import TensorFlow
import Batcher1

// Base use
// Some raw items (for instance filenames)
let rawItems = 0..<512
// A heavy-compute function lazily mapped on it (for instance, opening the images)
let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }
// A `Batcher` defined on this:
var batcher = Batcher(on: dataSet, batchSize: 64)
// Iteration over it (for instance, on epoch of the training loop) 
batcher.reorder(shuffled: true)
for batch in batcher {
    print(batch.shape)
}

// Use with padding
// Let's create an array of things of various lengths (for instance texts)
var dataSet: [Tensor<Int32>] = []
for _ in 0..<512 {
    dataSet.append(Tensor<Int32>(
        randomUniform: [Int.random(in: 1...200)], 
        lowerBound: Tensor<Int32>(0), 
        upperBound: Tensor<Int32>(100)
    ))
}

// We need to pad those tensors to make them all the same length.
// We could do this in one lazy transform applied beforehand and pad everything
// to the same length, but it's not memory-efficient: some batches might need less
// padding. So we need to add the padding after having selected the samples we
// are trying to batch.
func padTensors(tensors: [Tensor<Int32>]) -> [Tensor<Int32>] {
    let maxLength = tensors.map{ $0.shape[0] }.max()!
    return tensors.map { (t: Tensor<Int32>) -> Tensor<Int32> in 
        let remaining = Tensor<Int32>(zeros: [maxLength - t.shape[0]])
        return Tensor<Int32>(concatenating: [t, remaining])
    }
} 

var batcher = Batcher(on: dataSet, batchSize: 64, padSamples: padTensors)
batcher.reorder()
for b in batcher {
    print(b.shape)
}

// Use with a sampler
// In our previous example, another way to be memory efficient is to batch
// samples of roughly the same lengths.
func sortSamples(on dataset: [Tensor<Int32>]) -> [Int] {
    return Array(0..<dataset.count).sorted { dataset[$0].shape[0] > dataset[$1].shape[0] }
}

// Just giving a quick example, but there should be a separate function to shuffle a bit 
// the sorted thing
var batcher = Batcher(on: dataSet, batchSize: 64, sampleIndices: sortSamples, padSamples: padTensors)
batcher.reorder()
for b in batcher {
    print(b.shape)
}

// Sometimes the shuffle method needs to be applied on the dataset itself, 
// like for language modeling. Here is a base version of that to test
// the API allows it.
struct DataSet: RandomAccessCollection {
    typealias Index = Int
    typealias Element = Tensor<Int32>
    
    let numbers: [[Int]]
    let sequenceLength: Int
    // The texts all concatenated together
    private var stream: [Int]
    
    var startIndex: Int { return 0 }
    var endIndex: Int { return stream.count / sequenceLength }
    func index(after i: Int) -> Int { i+1 }
    
    init(numbers: [[Int]], sequenceLength: Int) {
        self.numbers = numbers
        self.sequenceLength = sequenceLength
        stream = numbers.reduce([], +)
    }
    
    subscript(index: Int) -> Tensor<Int32> {
        get { 
            let i = index * sequenceLength
            return Tensor<Int32>(stream[i..<i+sequenceLength].map { Int32($0)} )
        }
    }
    
    mutating func shuffle() {
        stream = numbers.shuffled().reduce([], +)
    }
}

//Let's create such a DataSet
let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
let dataset = DataSet(numbers: numbers, sequenceLength: 3)

// This is the shuffle function we will use
func internalShuffle(on dataset: inout DataSet, indices: [Int]) -> [Int] {
    dataset.shuffle()
    return indices
}

//Now let's look at what it gives us:
var batcher = Batcher(on: dataset, batchSize: 3, shuffleIndices: internalShuffle)
batcher.reorder()
for b in batcher {
    print(b)
}

batcher.reorder(shuffled: true)
for b in batcher {
    print(b)
}