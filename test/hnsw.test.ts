import HNSW from "../src/index"
import { readNdjson } from "./utils"
import { readFileSync } from "fs"
import { join } from "path"

describe("HNSW", () => {
  let hnsw: HNSW
  const dummyData = join(__dirname, "dummy-data")

  beforeEach(() => {
    hnsw = new HNSW()
  })

  describe("constructor", () => {
    it("should initialize with default values", () => {
      expect(hnsw.getSize()).toBe(0)
      expect(hnsw.getEntryPoint()).toBeNull()
    })

    it("should accept custom values", () => {
      const customHnsw = new HNSW(10, 100)
      expect(customHnsw.getSize()).toBe(0)
      expect(customHnsw.getEntryPoint()).toBeNull()
    })
  })

  describe("addNode", () => {
    it("should add a node", () => {
      hnsw.addNode(1, [1, 2])
      expect(hnsw.getSize()).toBe(1)
      expect(hnsw.getEntryPoint()).toBe(1)
    })

    it("should contain the correct number of vectors when reading from a known index", async () => {
      const embeddings = readNdjson(join(dummyData, "embeddings.test.ndjson"))
      const index = new HNSW()

      for (const [nodeId, vector] of Object.entries(embeddings)) {
        index.addNode(parseInt(nodeId), vector)
      }

      expect(index.getSize()).toBe(420)
    })
  })

  describe("deleteNode", () => {
    it("should delete a node", () => {
      hnsw.addNode(1, [1, 2])
      hnsw.deleteNode(1)
      expect(hnsw.getSize()).toBe(0)
    })

    it("should handle deletion of non-existent node", () => {
      // Capture console output
      const consoleOutput: string[] = []
      console.log = (output: string) => consoleOutput.push(output)

      hnsw.deleteNode(1)
      expect(consoleOutput.length).toBe(1)
      expect(consoleOutput[0]).toBe("Node with ID 1 not found.")
    })
  })

  describe("search", () => {
    it("should return an empty array if no nodes have been added", () => {
      const result = hnsw.search([1, 2])
      expect(result).toEqual([])
    })
  })

  describe("serialize & deserialize", () => {
    it("should serialize and deserialize correctly", () => {
      hnsw.addNode(1, [1, 2])
      const serialized = hnsw.serialize()
      const deserializedHnsw = HNSW.deserialize(serialized)

      expect(deserializedHnsw.getSize()).toBe(1)
      expect(deserializedHnsw.getEntryPoint()).toBe(1)
    })

    it("should have the same results after deserializing the index", async () => {
      const embeddings = readNdjson(join(dummyData, "embeddings.test.ndjson"))
      const queryVector = JSON.parse(readFileSync(join(dummyData, "prompt.test.ndjson"), "utf8").split("\n")[0])["vector"] as number[]
      const cleanIndex = new HNSW()

      for (const [nodeId, vector] of Object.entries(embeddings)) {
        cleanIndex.addNode(parseInt(nodeId), vector)
      }

      const cleanResults = cleanIndex.search(queryVector, 10)
      expect(cleanResults.length).toBe(10)

      const serialized = cleanIndex.serialize()

      const deserializedIndex = HNSW.deserialize(serialized)

      const deserializedResults = deserializedIndex.search(queryVector, 10)
      expect(deserializedResults.length).toBe(10)

      for (const [index, [cleanScore, cleanNodeId]] of cleanResults.entries()) {
        const [deserializedScore, deserializedNodeId] = deserializedResults[index]
        expect(cleanScore).toBeCloseTo(deserializedScore, 4)
        expect(cleanNodeId).toBe(deserializedNodeId)
      }
    })
  })

  describe("clear", () => {
    it("should clear all nodes", () => {
      hnsw.addNode(1, [1, 2])
      hnsw.clear()
      expect(hnsw.getSize()).toBe(0)
      expect(hnsw.getEntryPoint()).toBeNull()
    })
  })
})
