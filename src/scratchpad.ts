import { HNSW } from "./index";
import { readFileSync } from "fs";

(async () => {
    const data = readFileSync('./src/base.embeddings.bin')
    const index = HNSW.deserialize(data)
    console.log(index.getSize())
    console.log(index.getNodes().next().value.vector)
})()