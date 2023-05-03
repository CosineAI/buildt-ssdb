import { HNSW } from '../src/index'; // Adjust the import path according to your project structure

// Helper function to generate random vectors for testing
function generateRandomVector(dimensions: number): Float32Array {
  const vector = new Float32Array(dimensions);
  for (let i = 0; i < dimensions; i++) {
    vector[i] = Math.random() * 2 - 1; // Generate a random value between -1 and 1
  }
  return vector;
}

describe('HNSW', () => {
  const numDimensions = 5;
  const M = 16;
  const ef = 50;

  test('should initialize and add nodes', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);

    const nodeId = 0;
    const vector = generateRandomVector(numDimensions);
    hnsw.addNode(nodeId, vector);

    expect(hnsw.getSize()).toBe(1);
    expect(hnsw.getNodeById(nodeId)).toBeDefined();
    expect(hnsw.getNodeById(nodeId)?.vector).toEqual(vector);
  });

  test('should delete nodes', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);

    const nodeId = 0;
    const vector = generateRandomVector(numDimensions);
    hnsw.addNode(nodeId, vector);
    hnsw.deleteNode(nodeId);

    expect(hnsw.getSize()).toBe(0);
    expect(hnsw.getNodeById(nodeId)).toBeUndefined();
  });

  test('should find nearest neighbors', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);

    const numNodes = 10;
    for (let i = 0; i < numNodes; i++) {
      const vector = generateRandomVector(numDimensions);
      hnsw.addNode(i, vector);
    }

    const queryVector = generateRandomVector(numDimensions);
    const k = 5;
    const nearestNeighbors = hnsw.searchKNN(queryVector, k);

    expect(nearestNeighbors.length).toBe(k);
  });

  test('should serialize and deserialize correctly', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);

    const numNodes = 10;
    for (let i = 0; i < numNodes; i++) {
      const vector = generateRandomVector(numDimensions);
      hnsw.addNode(i, vector);
    }

    const serializedData = hnsw.serialize();
    const deserializedHNSW = HNSW.deserialize(serializedData);

    const queryVector = generateRandomVector(numDimensions);
    const k = 5;
    const nearestNeighbors = hnsw.searchKNN(queryVector, k);
    const deserializedNearestNeighbors = deserializedHNSW.searchKNN(queryVector, k);

    expect(nearestNeighbors.map(n => n.id)).toEqual(deserializedNearestNeighbors.map(n => n.id));
  });

  test('should get node by id', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);
  
    const nodeId = 0;
    const vector = generateRandomVector(numDimensions);
    hnsw.addNode(nodeId, vector);
  
    const node = hnsw.getNodeById(nodeId);
  
    expect(node).toBeDefined();
    expect(node!.id).toBe(nodeId);
    expect(node!.vector).toEqual(vector);
  });
  
  test('should get the total size of the index', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);
  
    const numNodes = 10;
    for (let i = 0; i < numNodes; i++) {
      const vector = generateRandomVector(numDimensions);
      hnsw.addNode(i, vector);
    }
  
    const size = hnsw.getSize();
  
    expect(size).toBe(numNodes);
  });
  
  test('should clear the index', () => {
    const hnsw = new HNSW('cosine', numDimensions, M, ef);
  
    const numNodes = 10;
    for (let i = 0; i < numNodes; i++) {
      const vector = generateRandomVector(numDimensions);
      hnsw.addNode(i, vector);
    }
  
    hnsw.clear();
  
    expect(hnsw.getSize()).toBe(0);
    expect(hnsw.getEntryPoint()).toBeNull();
    expect(hnsw.getMaxLevel()).toBe(0);
  });
  

});
