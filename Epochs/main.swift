import TensorFlow

// Base use
// Some raw items (for instance filenames)
let rawItems = 0..<512

do { 
  // A heavy-compute function lazily mapped on it (for instance, opening the images)
  let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }
  
  // A `Batches` defined on this:
  let batches = Batches(of: 64, from: dataSet, \.collated)
  // Iteration over it:
  for batch in batches {
    print(batch.shape)
  }

  // Enabling shuffle
  let batches2 = Batches(of: 64, from: dataSet.shuffled(), \.collated)
  print(batches2.map(\.shape))
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

do {
  let batches = Batches(of: 64, from: dataSet, tailPaddedWith0AndCollated)
  for b in batches {
    print(b.shape)
  }
}

// Use with a sampler
do {
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  let sortedDataset = dataSet.sorted { $0.shape[0] > $1.shape[0] }

  let batches = Batches(of: 64, from: sortedDataset, tailPaddedWith0AndCollated)
  for b in batches {
    print(b.shape)
  }
}

do {
  // When using a `batchSize` we get a bit of shuffle:
  let sortedDataset = dataSet.shuffled().sortedInBatches(of: 256) {
    $0.shape[0] > $1.shape[0]
  }

  let batches = Batches(of: 64, from: sortedDataset, tailPaddedWith0AndCollated)
  for b in batches {
    print(b.shape)
  }
}

struct LanguageModelDataset<Texts: RandomAccessCollection> where Texts.Element == [Int] {
    /// The underlying collection of texts
    public var texts: Texts
    /// The length of the samples returned when indexing
    private let sequenceLength: Int
    // The texts all concatenated together
    private var stream: [Int]
    
    init(texts: Texts, sequenceLength: Int) {
        self.texts = texts
        self.sequenceLength = sequenceLength
        stream = texts.reduce([], +)
    }
}

extension LanguageModelDataset: RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = Tensor<Int32>
    
    public var startIndex: Int { return 0 }
    public var endIndex: Int { return stream.count / sequenceLength }
    public func index(after i: Int) -> Int { i+1 }
    
    public subscript(index: Int) -> Tensor<Int32> {
        get { 
            let i = index * sequenceLength
            return Tensor<Int32>(stream[i..<i+sequenceLength].map { Int32($0)} )
        }
    }
}

//Let's create such a DataSet
let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
do {
  let dataset = LanguageModelDataset(texts: numbers, sequenceLength: 3)

  //Now let's look at what it gives us:
  let batches = Batches(of: 3, from: dataset, \.collated)
  for b in batches {
    print(b)
  }
}

do {
  let dataset = LanguageModelDataset(texts: numbers.shuffled(), sequenceLength: 3)
  let batches = Batches(of: 3, from: dataset, \.collated)
  for b in batches {
    print(b)
  }
}

