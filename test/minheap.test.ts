import { MinHeap } from "../src/minheap"

describe("MinHeap", () => {
  test("should create an empty heap", () => {
    const heap = new MinHeap<number>((a, b) => a - b)
    expect(heap.size()).toBe(0)
  })

  test("should push values into the heap", () => {
    const heap = new MinHeap<number>((a, b) => a - b)
    heap.push(3)
    heap.push(2)
    heap.push(1)
    expect(heap.size()).toBe(3)
  })

  test("should pop the minimum value from the heap", () => {
    const heap = new MinHeap<number>((a, b) => a - b)
    heap.push(3)
    heap.push(2)
    heap.push(1)
    expect(heap.pop()).toBe(1)
  })

  test("should maintain the min-heap property", () => {
    const heap = new MinHeap<number>((a, b) => a - b)
    heap.push(5)
    heap.push(2)
    heap.push(8)
    heap.push(4)
    heap.push(1)
    expect(heap.pop()).toBe(1)
    expect(heap.pop()).toBe(2)
    expect(heap.pop()).toBe(4)
    expect(heap.pop()).toBe(5)
    expect(heap.pop()).toBe(8)
  })

  test("should return undefined when popping from an empty heap", () => {
    const heap = new MinHeap<number>((a, b) => a - b)
    expect(heap.pop()).toBeUndefined()
  })

  test("should handle custom comparison functions", () => {
    interface Person {
      name: string
      age: number
    }

    const peopleHeap = new MinHeap<Person>((a, b) => a.age - b.age)

    peopleHeap.push({ name: "Alice", age: 28 })
    peopleHeap.push({ name: "Bob", age: 25 })
    peopleHeap.push({ name: "Charlie", age: 30 })

    expect(peopleHeap.pop()?.name).toBe("Bob")
    expect(peopleHeap.pop()?.name).toBe("Alice")
    expect(peopleHeap.pop()?.name).toBe("Charlie")
  })
})
