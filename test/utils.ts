import * as fs from "fs"

export function readNdjson(filePath: string): { [id: string]: number[] } {
  const data: { [id: string]: number[] } = {}
  const fileContents = fs.readFileSync(filePath, "utf8")
  const lines = fileContents.split("\n")

  for (const line of lines) {
    if (line.trim() === "") continue
    const item = JSON.parse(line)
    const nodeId = String(item["node_id"])
    const vector = item["vector"] as number[]
    data[nodeId] = vector
  }

  return data
}
