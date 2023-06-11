export class MinHeap<T> {
  private heap: T[]
  private compare: (a: T, b: T) => number

  constructor(compare: (a: T, b: T) => number) {
    this.heap = []
    this.compare = compare
  }

  size() {
    return this.heap.length
  }

  peek(): T | undefined {
    return this.heap.length > 0 ? this.heap[0] : undefined
  }

  push(val: T) {
    this.heap.push(val)
    let idx = this.heap.length - 1
    while (idx > 0) {
      const parentIdx = Math.floor((idx - 1) / 2)
      if (this.compare(this.heap[idx], this.heap[parentIdx]) < 0) {
        // eslint-disable-next-line @typescript-eslint/no-extra-semi
        ;[this.heap[idx], this.heap[parentIdx]] = [this.heap[parentIdx], this.heap[idx]]
        idx = parentIdx
      } else {
        break
      }
    }
  }

  pop(): T | undefined {
    if (this.heap.length === 0) {
      return undefined
    }
    const result = this.heap[0]
    const end = this.heap.pop()
    if (this.heap.length > 0 && end) {
      this.heap[0] = end
      let idx = 0
      const length = this.heap.length
      const element = this.heap[0]
      while (true) {
        const leftChildIdx = 2 * idx + 1
        const rightChildIdx = 2 * idx + 2
        let leftChild: T | null = null
        let rightChild: T | null = null
        let swapIdx: number | null = null

        if (leftChildIdx < length) {
          leftChild = this.heap[leftChildIdx]
          if (this.compare(leftChild, element) < 0) {
            swapIdx = leftChildIdx
          }
        }
        if (rightChildIdx < length) {
          rightChild = this.heap[rightChildIdx]
          if (
            (swapIdx === null && this.compare(rightChild, element) < 0) ||
            (leftChild !== null && swapIdx !== null && this.compare(rightChild, leftChild as T) < 0)
          ) {
            swapIdx = rightChildIdx
          }
        }
        if (swapIdx === null) break
        this.heap[idx] = this.heap[swapIdx]
        this.heap[swapIdx] = element
        idx = swapIdx
      }
    }
    return result
  }
}
