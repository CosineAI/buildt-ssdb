/**
 * Authors: Alistair Pullen, GPT-4
 */

export type SimilarityMetric = 'cosine' | 'euclidean';

export class HNSWNode {
  id: number;
  vector: Float32Array;
  level: number;
  neighbors: HNSWNode[][];

  constructor(id: number, vector: Float32Array, level: number, M: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = new Array(level + 1).fill(null).map(() => new Array(M).fill(null));
  }
}

export class HNSW {
  
  private similarityMetric: SimilarityMetric;
  private numDimensions: number;
  private M: number;
  private ef: number;
  private maxLevel: number;
  private entryPoint: HNSWNode | null;
  private nodes: Map<number, HNSWNode>;

  constructor(similarityMetric: SimilarityMetric, numDimensions: number, M: number, ef: number) {
    this.similarityMetric = similarityMetric;
    this.numDimensions = numDimensions;
    this.M = M;
    this.ef = ef;
    this.maxLevel = 0;
    this.entryPoint = null;
    this.nodes = new Map();
  }

  private randomLevel(): number {
    let level = 0;
    while (Math.random() < 1 / Math.E && level < this.maxLevel) {
      level++;
    }
    return level;
  }

  private computeDistance(a: Float32Array, b: Float32Array): number {
    let distance = 0;
    if (this.similarityMetric === 'euclidean') {
      for (let i = 0; i < this.numDimensions; i++) {
        distance += Math.pow(a[i] - b[i], 2);
      }
      return Math.sqrt(distance);
    } else if (this.similarityMetric === 'cosine') {
      let dotProduct = 0;
      let normA = 0;
      let normB = 0;
      for (let i = 0; i < this.numDimensions; i++) {
        dotProduct += a[i] * b[i];
        normA += Math.pow(a[i], 2);
        normB += Math.pow(b[i], 2);
      }
      return 1 - dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    } else {
      throw new Error('Invalid similarity metric');
    }
  }

  private searchLevel(query: Float32Array, entryPoint: HNSWNode, level: number): HNSWNode {
    let current = entryPoint;
    let bestDistance = this.computeDistance(query, current.vector);
    let changed = true;

    while (changed) {
      changed = false;
      for (const neighbor of current.neighbors[level]) {
        if (!neighbor) {
          break;
        }
        const distance = this.computeDistance(query, neighbor.vector);
        if (distance < bestDistance) {
          bestDistance = distance;
          current = neighbor;
          changed = true;
        }
      }
    }

    return current;
  }

  public addNode(id: number, vector: Float32Array): void {
    const level = this.randomLevel();
    const newNode = new HNSWNode(id, vector, level, this.M);

    if (!this.entryPoint) {
      this.entryPoint = newNode;
      this.nodes.set(id, newNode);
      return;
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
    }

    let currentNode = this.searchLevel(vector, this.entryPoint, this.maxLevel);

    for (let i = this.maxLevel; i >= 0; i--) {
      if (i <= level) {
        const neighbors = this._searchKNN(currentNode, vector, this.M, i);
        newNode.neighbors[i] = neighbors.slice(0, this.M);

        for (const neighbor of neighbors) {
          const nDist = this.computeDistance(neighbor.vector, vector);
          const furthestNeighborIndex = neighbor.neighbors[i].findIndex(n => n === null || nDist < this.computeDistance(neighbor.vector, n.vector));
          if (furthestNeighborIndex !== -1) {
            neighbor.neighbors[i].splice(furthestNeighborIndex, 0, newNode);
            if (neighbor.neighbors[i].length > this.M) {
              neighbor.neighbors[i].pop();
            }
          }
        }
      }

      if (i > 0) {
        currentNode = this.searchLevel(vector, currentNode, i - 1);
      }
    }

    this.nodes.set(id, newNode);
  }

  private _searchKNN(entryPoint: HNSWNode, query: Float32Array, k: number, level: number): HNSWNode[] {
    const visited = new Set<number>();
    const candidates = new Set<HNSWNode>([entryPoint]);
    const result = new Array<HNSWNode>();

    while (candidates.size > 0) {
      const closest = Array.from(candidates).reduce((best, curr) => {
        const bestDist = best ? this.computeDistance(query, best.vector) : Infinity;
        const currDist = this.computeDistance(query, curr.vector);
        return currDist < bestDist ? curr : best;
      }, null as HNSWNode | null);

      if (!closest) {
        break;
      }

      candidates.delete(closest);
      visited.add(closest.id);

      if (result.length < k) {
        result.push(closest);
      } else {
        const furthestResult = result.reduce((furthest, curr) => {
          const furthestDist = furthest ? this.computeDistance(query, furthest.vector) : -Infinity;
          const currDist = this.computeDistance(query, curr.vector);
          return currDist > furthestDist ? curr : furthest;
        }, null as HNSWNode | null);

        if (!furthestResult) {
          break;
        }

        const closestDist = this.computeDistance(query, closest.vector);
        const furthestResultDist = this.computeDistance(query, furthestResult.vector);

        if (closestDist < furthestResultDist) {
          result.splice(result.indexOf(furthestResult), 1, closest);
        } else {
          break;
        }
      }

      for (const neighbor of closest.neighbors[level]) {
        if (!neighbor || visited.has(neighbor.id)) {
          continue;
        }
        candidates.add(neighbor);
      }
    }

    return result;
  }

  public searchKNN(query: Float32Array, k: number): HNSWNode[] {
    if (!this.entryPoint) {
      return [];
    }
    const entryPoint = this.searchLevel(query, this.entryPoint, this.maxLevel);
    return this._searchKNN(entryPoint, query, k, 0);
  }

  public deleteNode(id: number): void {
    const nodeToDelete = this.nodes.get(id);
    if (!nodeToDelete) {
      console.warn(`Node with ID ${id} not found. Skipping delete.`);
      return;
    }

    for (let level = 0; level <= nodeToDelete.level; level++) {
      const neighbors = nodeToDelete.neighbors[level];
      for (const neighbor of neighbors) {
        if (!neighbor) {
          continue;
        }
        const index = neighbor.neighbors[level].findIndex(n => n && n.id === id);
        if (index !== -1) {
          neighbor.neighbors[level].splice(index, 1);
          neighbor.neighbors[level].push(undefined as any);
        }
      }
    }

    if (this.entryPoint && this.entryPoint.id === id) {
      this.entryPoint = null;
      this.maxLevel = 0;
      for (const node of this.nodes.values()) {
        if (node.level > this.maxLevel) {
          this.entryPoint = node;
          this.maxLevel = node.level;
        }
      }
    }

    this.nodes.delete(id);
  }

  public serialize(): Uint8Array {
    const nodeCount = this.nodes.size;
    const headerSize = 1 + 4 * 4;
    const nodeSize = 4 + this.numDimensions * 4 + 4 + this.M * 4 * (this.maxLevel + 1);
    const bufferSize = headerSize + nodeCount * nodeSize;

    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);

    let offset = 0;

    view.setUint8(offset, this.similarityMetric === 'cosine' ? 0 : 1);
    offset += 1;

    view.setUint32(offset, this.numDimensions, true);
    offset += 4;

    view.setUint32(offset, this.M, true);
    offset += 4;

    view.setUint32(offset, this.ef, true);
    offset += 4;

    view.setUint32(offset, this.entryPoint ? this.entryPoint.id : -1, true);
    offset += 4;

    for (const node of this.nodes.values()) {
      view.setUint32(offset, node.id, true);
      offset += 4;

      for (let i = 0; i < this.numDimensions; i++) {
        view.setFloat32(offset, node.vector[i], true);
        offset += 4;
      }

      view.setUint32(offset, node.level, true);
      offset += 4;

      for (let i = 0; i <= this.maxLevel; i++) {
        for (let j = 0; j < this.M; j++) {
          const neighbor = node.neighbors[i][j];
          view.setUint32(offset, neighbor ? neighbor.id : -1, true);
          offset += 4;
        }
      }
    }

    return new Uint8Array(buffer);
  }

  public static deserialize(data: Uint8Array): HNSW {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

    let offset = 0;

    const similarityMetric = view.getUint8(offset) === 0 ? 'cosine' : 'euclidean';
    offset += 1;

    const numDimensions = view.getUint32(offset, true);
    offset += 4;

    const M = view.getUint32(offset, true);
    offset += 4;

    const ef = view.getUint32(offset, true);
    offset += 4;

    const entryPointId = view.getUint32(offset, true);
    offset += 4;

    const hnsw = new HNSW(similarityMetric, numDimensions, M, ef);

    // Temporary storage for neighbor IDs
    const tempNeighborIds: Map<number, number[][]> = new Map();

    while (offset < data.byteLength) {
      const id = view.getUint32(offset, true);
      offset += 4;

      const vector = new Float32Array(numDimensions);
      for (let i = 0; i < numDimensions; i++) {
        vector[i] = view.getFloat32(offset, true);
        offset += 4;
      }

      const level = view.getUint32(offset, true);
      offset += 4;

      const node = new HNSWNode(id, vector, level, M);

      const neighborIds: number[][] = [];
      for (let i = 0; i <= hnsw.maxLevel; i++) {
        neighborIds[i] = [];
        for (let j = 0; j < M; j++) {
          const neighborId = view.getUint32(offset, true);
          offset += 4;
          neighborIds[i].push(neighborId);
        }
      }

      tempNeighborIds.set(id, neighborIds);

      hnsw.nodes.set(id, node);
      if (id === entryPointId) {
        hnsw.entryPoint = node;
        hnsw.maxLevel = level;
      }
    }

    // Assign neighbors after all nodes have been created
    tempNeighborIds.forEach((neighborIds, nodeId) => {
      const node = hnsw.nodes.get(nodeId);
      if (node) {
        neighborIds.forEach((ids, level) => {
          ids.forEach((id, j) => {
            if (id !== -1 && hnsw.nodes.has(id)) {
              node.neighbors[level][j] = hnsw.nodes.get(id)!;
            }
          });
        });
      }
    });

    return hnsw;
  }

  public getNodeById(id: number): HNSWNode | undefined {
    return this.nodes.get(id);
  }

  public getSize(): number {
    return this.nodes.size;
  }

  public getEntryPoint(): HNSWNode | null {
    return this.entryPoint;
  }

  public getMaxLevel(): number {
    return this.maxLevel;
  }

  public clear(): void {
    this.nodes = new Map();
    this.entryPoint = null;
    this.maxLevel = 0;
  }

}
