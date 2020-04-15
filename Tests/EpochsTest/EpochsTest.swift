import TensorFlow
//import Epochs
//import XCTest

func XCTAssert(
  _ expression: @autoclosure () -> Bool,
  _ message: @autoclosure () -> String = "",
  file: StaticString = #file, line: UInt = #line
) {
  assert(expression(), message(), file: file, line: line)
}

func XCTAssertFalse(
  _ expression: @autoclosure () -> Bool,
  _ message: @autoclosure () -> String = "",
  file: StaticString = #file, line: UInt = #line
) {
  assert(!expression(), message(), file: file, line: line)
}

func XCTAssertEqual<T: Equatable>(
  _ expression: @autoclosure () -> T,
  _ expression1: @autoclosure () -> T,
  _ message: @autoclosure () -> String = "",
  file: StaticString = #file, line: UInt = #line
) {
  assert(expression() == expression1(), message(), file: file, line: line)
}

var pcg = LinearCongruential(seed: 42)
let tfSeed: TensorFlowSeed = (
  graph: Int32.random(in:Int32.min..<Int32.max, using: &pcg), 
  op: Int32.random(in:Int32.min..<Int32.max, using: &pcg))

class Box {
  var accessed: Bool = false
}

final class EpochsTests /*: XCTestCase*/ {
  // Some raw items (for instance filenames)
  let rawItems: [Box] = Array(0..<512).map{ _ in Box() }
  
  func testLazyShuffle() {
    // A lazy dataset and an array that keeps track of whether the elements were
    // accessed or not.
    let dataset = rawItems.lazy.map { (x: Box) -> Tensor<Float> in
      x.accessed = true
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    
    // Using `.shuffled()` access all elements
    let _ = dataset.shuffled(using: &pcg)
    XCTAssert(rawItems.allSatisfy { $0.accessed }) 

    // Using `.innerShuffled()` on a `ReindexedColletion` does not access elements
    let _ = rawItems.map { $0.accessed = false }
    let _ = ReindexedCollection(dataset).innerShuffled(using: &pcg)
    XCTAssert(rawItems.allSatisfy { !$0.accessed }) 
  }
  
  func testBaseUse() {
   // A heavy-compute function lazily mapped on the raw items (for instance, opening the images)
    let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3], seed: tfSeed) }

    // A `Batcher` defined on this:
    let batches = Batches(of: 64, from: dataSet, \.collated)
    // Iteration over it (for instance, on epoch of the training loop) 
    XCTAssert(batches.allSatisfy { 
      $0.shape == TensorShape([64, 224, 224, 3]) }
    )
  }
    
  // Tests with shuffle
  func testShuffle() {
    let dataset = rawItems.shuffled().lazy.map { (x: Box) -> Tensor<Float> in
      x.accessed = true
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }

    let batches = Batches(of: 64, from: dataset, \.collated)
    for (i, batch) in batches.enumerated() {
      XCTAssertEqual(batch.shape, TensorShape([64, 224, 224, 3]))
      // Test randomness.
      if i == 0 {
        XCTAssertFalse(rawItems[0..<64].allSatisfy { $0.accessed })
      }
      // All those tests true will mean we accessed every element of the dataset
      // exactly once
      XCTAssertEqual(rawItems.filter() { $0.accessed }.count, (i + 1) * 64)
    }
  }
  
  func testInnerShuffle() {
    let dataset = rawItems.lazy.map { (x: Box) -> Tensor<Float> in
      x.accessed = true
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }

    let batches = Batches(of: 64, from:  ReindexedCollection(dataset).innerShuffled(using: &pcg), \.collated)
    for (i, batch) in batches.enumerated() {
      XCTAssertEqual(batch.shape, TensorShape([64, 224, 224, 3]))
      // Test randomness.
      if i == 0 {
        XCTAssertFalse(rawItems[0..<64].allSatisfy { $0.accessed })
      }
      // All those tests true will mean we accessed every element of the dataset
      // exactly once
      XCTAssertEqual(rawItems.filter() { $0.accessed }.count, (i + 1) * 64)
    }
  }
    
  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let dataSet: [Tensor<Int32>] = { 
    var dataSet: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataSet.append(Tensor<Int32>(
                       randomUniform: [Int.random(in: 1...200, using: &pcg)], 
                       lowerBound: Tensor<Int32>(0), 
                       upperBound: Tensor<Int32>(100),
                       seed: tfSeed
                    ))
    }
    return dataSet
  }()
    
  func paddingTest(padValue: Int32, padFirst: Bool) {
    let batches = Batches(of: 64, from: dataSet) { $0.paddedAndCollated(with: padValue) }
    for (i, b) in batches.enumerated() {
      let shapes = dataSet[(i * 64)..<((i + 1) * 64)].map { Int($0.shape[0]) }
      let expectedShape = shapes.reduce(0) { max($0, $1) }
      XCTAssertEqual(Int(b.shape[1]), expectedShape)
        
      for k in 0..<64 {
        let currentShape = dataSet[i * 64 + k].shape[0]
        XCTAssertEqual(
          b[k, currentShape..<expectedShape], 
          Tensor<Int32>(repeating: padValue, shape: [expectedShape - currentShape]))         
      }
    }
  }

  func testAllPadding() {
    paddingTest(padValue: 0, padFirst: false)
    paddingTest(padValue: 42, padFirst: false)
    paddingTest(padValue: 0, padFirst: true)
    paddingTest(padValue: -1, padFirst: true)
  }

  // Use with a sampler
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  func testSortAndPadding() {
    // Use with a sampler
    // In our previous example, another way to be memory efficient is to batch
   // samples of roughly the same lengths.
    let sortedDataset = dataSet.sorted { $0.shape[0] > $1.shape[0] }
      
    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    var previousSize: Int? = nil
    for batch in batches {
      if let size = previousSize {
        XCTAssert(size >= batch.shape[1])
      }
      previousSize = Int(batch.shape[1])
    }
  }
    
  func testSortishAndPadding() {
    // When using a `batchSize` we get a bit of shuffle:
    // This can all be applied on a lazy collection without breaking the lasziness as long as the sort function does not access the dataset
    let sortedDataset = ReindexedCollection(dataSet).innerShuffled(using: &pcg).sortedInBatches(of: 256) { 
      dataSet[$0].shape[0] > dataSet[$1].shape[0] 
    }

    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    var previousSize: Int? = nil
    for (i, batch) in batches.enumerated() {
      if let size = previousSize {
        XCTAssert(i%4 != 0 ? size >= batch.shape[1] : size <= batch.shape[1])
      }
      previousSize = Int(batch.shape[1])
    }
  }
    
  struct LanguageModelDataset<Texts: RandomAccessCollection>: Collection where Texts.Element == [Int] {
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
    
  //Now let's look at what it gives us:
  func testLanguageModel() {
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers, sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    for (i, batch) in batches.enumerated() {
      let expected = Tensor<Int32>(rangeFrom: Int32(1 + i * 9), to: Int32(1 + (i + 1) * 9), stride: 1)
      XCTAssertEqual(batch, expected.reshaped(to: [3, 3]))
    }
  }
  
  func isSubset(_ x: [Int], from y: [Int]) -> Bool {
    if let i = y.firstIndex(of: x[0]) {
      return x.enumerated().allSatisfy() { (k: Int, o: Int) -> Bool in
        o == y[i + k]
      }  
    }
    return false
  }

  func testLanguageModelShuffled() {
    // To shuffle it we need to go back to the inner numbers
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers.shuffled(using: &pcg), sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    
    var stream: [Int] = []
    for batch in batches {
      stream += batch.scalars.map { Int($0) }
    }
      
    // This checks the stream contains all texts once.
    XCTAssertEqual(stream.count, 18)
    XCTAssert(numbers.allSatisfy{ isSubset($0, from: stream) })
  }
}


extension EpochsTests {
  static var allTests = [
    ("testLazyShuffle", testLazyShuffle),
    ("testBaseUse", testBaseUse),
    ("testShuffle", testShuffle),
    ("testInnerShuffle", testInnerShuffle),
    ("testAllPadding", testAllPadding),
    ("testSortAndPadding", testSortAndPadding),
    ("testSortishAndPadding", testSortishAndPadding),
    ("testLanguageModel", testLanguageModel),
    ("testLanguageModelShuffled", testLanguageModelShuffled),
  ]
}

