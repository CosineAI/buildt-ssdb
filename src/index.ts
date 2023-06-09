import { MinHeap } from "./minheap"

export class Node {
  public node_id: number
  public vector: number[]
  public level: number | null
  public neighbors: Set<number>

  constructor(node_id: number, vector: number[]) {
    this.node_id = node_id
    this.vector = vector
    this.level = null
    this.neighbors = new Set()
  }

  public addNeighbor(node_id: number) {
    this.neighbors.add(node_id)
  }
}

export default class HNSW {
  private M: number
  private ef_construction: number
  private nodes: { [id: number]: Node }
  private levels: Node[][]
  private enter_point: number | null

  constructor(M = 16, ef_construction = 200) {
    this.M = M
    this.ef_construction = ef_construction
    this.nodes = {}
    this.levels = []
    this.enter_point = null
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i]
    }
    return dotProduct
  }

  private insertNode(node: Node, level: number): number {
    const node_id = node.node_id
    node.level = level
    this.nodes[node_id] = node
    while (level >= this.levels.length) {
      this.levels.push([])
    }
    this.levels[level].push(node)
    return node_id
  }

  public deleteNode(nodeId: number): void {
    if (!(nodeId in this.nodes)) {
      console.log(`Node with ID ${nodeId} not found.`)
      return
    }

    const node = this.nodes[nodeId]
    const level = node.level as number

    for (const neighborId of node.neighbors) {
      const neighbor = this.nodes[neighborId]
      neighbor.neighbors.delete(nodeId)
    }

    const indexInLevel = this.levels[level].findIndex(n => n.node_id === nodeId)
    if (indexInLevel !== -1) {
      this.levels[level].splice(indexInLevel, 1)
    }

    delete this.nodes[nodeId]

    if (this.levels[level].length === 0) {
      this.levels.splice(level, 1)
    }
  }

  private searchLayer(node_id: number, query: number[], ef: number): [number, number][] {
    const visited = new Set<number>()
    const candidates: [number, number][] = [[-1.0, node_id]]
    const bestCandidates: [number, number][] = []

    const minHeap = new MinHeap<[number, number]>((a, b) => a[0] - b[0])

    while (candidates.length > 0) {
      const [dist, curNodeId] = candidates.pop() as [number, number]
      if (!visited.has(curNodeId)) {
        visited.add(curNodeId)
        const curNode = this.nodes[curNodeId]

        if (bestCandidates.length < ef || dist > bestCandidates[0][0]) {
          minHeap.push([dist, curNodeId])
          if (minHeap.size() > ef) {
            minHeap.pop()
          }
        }

        if (curNode) {
          for (const neighborId of curNode.neighbors) {
            if (!visited.has(neighborId)) {
              const neighbor = this.nodes[neighborId]
              const dist = this.cosineSimilarity(query, neighbor.vector)
              candidates.push([dist, neighborId])
            }
          }
        } else {
          return []
        }
      }
    }

    while (minHeap.size() > 0) {
      bestCandidates.push(minHeap.pop() as [number, number])
    }

    return bestCandidates.sort((a, b) => b[0] - a[0])
  }

  public search(query: number[], ef?: number): [number, number][] {
    if (ef === undefined) {
      ef = this.ef_construction
    }

    let curNodeId = this.enter_point
    let curLevel = this.levels.length - 1

    while (curLevel > 0) {
      const candidates = this.searchLayer(curNodeId as number, query, 1)
      curNodeId = candidates[0][1]
      curLevel -= 1
    }

    return this.searchLayer(curNodeId as number, query, ef as number)
  }

  public addNode(node_id: number, vector: number[]) {
    if (this.enter_point === null) {
      const node = new Node(node_id, vector)
      this.insertNode(node, 0)
      this.enter_point = node_id
      return
    }

    const maxLevel = Math.floor(Math.log2(Object.keys(this.nodes).length)) + 1
    let curLevel = 0
    while (Math.random() < 0.5 && curLevel < maxLevel) {
      curLevel += 1
    }

    const node = new Node(node_id, vector)
    this.insertNode(node, curLevel)

    let curNodeId = this.enter_point
    curLevel = this.levels.length - 1

    while (curLevel > (node.level as number)) {
      const candidates = this.searchLayer(curNodeId, vector, 1)
      curNodeId = candidates[0][1]
      curLevel -= 1
    }

    while (curLevel >= 0) {
      const candidates = this.searchLayer(curNodeId, vector, this.M)
      curNodeId = candidates[0][1]

      if (candidates.length > this.M) {
        candidates.splice(0, candidates.length - this.M)
      }

      for (const [, neighborId] of candidates) {
        node.addNeighbor(neighborId)
        this.nodes[neighborId].addNeighbor(node_id)
      }

      curLevel -= 1
    }
  }

  public serialize(): Uint8Array {
    // Compute the size needed for the buffer
    let bufferSize = 16 // M (4 bytes) + ef_construction (4 bytes) + nodesCount (4 bytes) + enter_point (4 bytes)
    for (const nodeId in this.nodes) {
      const node = this.nodes[nodeId]
      bufferSize += 12 + node.vector.length * 4 + 4 + node.neighbors.size * 4
    }
    bufferSize += 4 // levelsCount (4 bytes)
    for (const level of this.levels) {
      bufferSize += 4 + level.length * 4
    }

    const buffer = new ArrayBuffer(bufferSize)
    const view = new DataView(buffer)
    let offset = 0

    // Serialize M, ef_construction, and enter_point
    view.setInt32(offset, this.M)
    offset += 4
    view.setInt32(offset, this.ef_construction)
    offset += 4
    view.setInt32(offset, this.enter_point !== null ? this.enter_point : -1)
    offset += 4

    // Serialize nodes
    const nodeIds = Object.keys(this.nodes).map(k => parseInt(k))
    view.setInt32(offset, nodeIds.length)
    offset += 4
    for (const nodeId of nodeIds) {
      const node = this.nodes[nodeId]
      view.setInt32(offset, node.node_id)
      offset += 4
      view.setInt32(offset, node.vector.length)
      offset += 4
      for (const value of node.vector) {
        view.setFloat32(offset, value)
        offset += 4
      }
      view.setInt32(offset, node.level as number)
      offset += 4
      view.setInt32(offset, node.neighbors.size)
      offset += 4
      for (const neighborId of node.neighbors) {
        view.setInt32(offset, neighborId)
        offset += 4
      }
    }

    // Serialize levels
    view.setInt32(offset, this.levels.length)
    offset += 4
    for (const level of this.levels) {
      view.setInt32(offset, level.length)
      offset += 4
      for (const node of level) {
        view.setInt32(offset, node.node_id)
        offset += 4
      }
    }

    return new Uint8Array(buffer)
  }

  public static deserialize(data: Uint8Array): HNSW {
    const buffer = data.buffer
    const view = new DataView(buffer)
    let offset = 0

    // Deserialize M, ef_construction, and enter_point
    const M = view.getInt32(offset)
    offset += 4
    const ef_construction = view.getInt32(offset)
    offset += 4
    const enter_point = view.getInt32(offset)
    offset += 4

    const hnsw = new HNSW(M, ef_construction)
    hnsw.enter_point = enter_point !== -1 ? enter_point : null

    // Deserialize nodes
    const nodesCount = view.getInt32(offset)
    offset += 4
    for (let i = 0; i < nodesCount; i++) {
      const node_id = view.getInt32(offset)
      offset += 4
      const vectorLen = view.getInt32(offset)
      offset += 4
      const vector: number[] = []
      for (let j = 0; j < vectorLen; j++) {
        vector.push(view.getFloat32(offset))
        offset += 4
      }
      const level = view.getInt32(offset)
      offset += 4
      const neighborsCount = view.getInt32(offset)
      offset += 4
      const neighbors = new Set<number>()
      for (let j = 0; j < neighborsCount; j++) {
        neighbors.add(view.getInt32(offset))
        offset += 4
      }

      const node = new Node(node_id, vector)
      node.level = level
      node.neighbors = neighbors
      hnsw.nodes[node_id] = node
    }

    // Deserialize levels
    const levelsCount = view.getInt32(offset)
    offset += 4
    for (let i = 0; i < levelsCount; i++) {
      const levelLen = view.getInt32(offset)
      offset += 4
      const level: Node[] = []
      for (let j = 0; j < levelLen; j++) {
        const nodeId = view.getInt32(offset)
        offset += 4
        level.push(hnsw.nodes[nodeId])
      }
      hnsw.levels.push(level)
    }

    return hnsw
  }

  public getSize(): number {
    return Object.keys(this.nodes).length
  }

  public getEntryPoint(): number | null {
    return this.enter_point
  }

  public clear(): void {
    this.nodes = {}
    this.levels = []
    this.enter_point = null
  }
}
