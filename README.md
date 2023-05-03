# buildt-ssdb: Super Search DataBase

`buildt-ssdb` is a simple, efficient, and easy-to-use implementation of the Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search. It's perfect for searching mid-scale, high-dimensional datasets quickly and with minimal memory overhead.

This package has no dependencies, so should be usable both in the browser and Node scenarios.

## Installation - NPM package not yet available

To install `buildt-ssdb`, run the following command in your project directory:

```sh
npm install buildt-ssdb
```

## Setup & Building
```sh
npm i
```
followed by
```sh
npm run build
```

## Usage

Here's an example of how to use buildt-ssdb:

```typescript
import { HNSW } from 'buildt-ssdb';

// Create an HNSW index with the cosine similarity metric, 5 dimensions, M = 16, and ef = 50
const hnsw = new HNSW('cosine', 5, 16, 50);

// Add nodes to the index
const nodeId1 = 0;
const vector1 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
hnsw.addNode(nodeId1, vector1);

const nodeId2 = 1;
const vector2 = new Float32Array([0.5, 0.4, 0.3, 0.2, 0.1]);
hnsw.addNode(nodeId2, vector2);

// Perform a k-nearest neighbor search with k = 3
const queryVector = new Float32Array([0.15, 0.25, 0.35, 0.45, 0.55]);
const nearestNeighbors = hnsw.searchKNN(queryVector, 3);

console.log('Nearest neighbors:', nearestNeighbors);
```

## Performance
The largest dataset I've pushed through this is a 100k vector index at 1536 dimensions per vector. The searchKNN method took ~2.36ms per query at this scale. I haven't gone any larger than this yet but likely will give it a try when I have the time.

## API
### HNSW
The main class for working with HNSW indices.

Constructor
```typescript
constructor(similarityMetric: 'cosine' | 'euclidean', numDimensions: number, M: number, ef: number)
```

### Methods
`addNode(id: number, vector: Float32Array): void`: Add a node to the index.
`deleteNode(id: number): void`: Delete a node from the index.
`searchKNN(queryVector: Float32Array, k: number): HNSWNode[]`: Perform a k-nearest neighbor search.
`getNodeById(id: number): HNSWNode | undefined`: Get a node by its ID.
`getSize(): number`: Get the total size of the index (number of nodes).
`serialize(): Uint8Array`: Serialize the index to a binary format.
`static deserialize(data: Uint8Array): HNSW`: Deserialize an index from its binary representation.

### Plans
I want to add capabilities for sharded indices, as that would be very useful for Buildt, as well as having a temporal component to the vector database – something I haven't yet seen from the main providers.