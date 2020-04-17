import TensorFlow
import Epochs
import XCTest
import PcgRandom

var pcg = Pcg64Random(seed: 42)
let tfSeed: TensorFlowSeed = (
  graph: Int32.random(in:Int32.min..<Int32.max, using: &pcg), 
  op: Int32.random(in:Int32.min..<Int32.max, using: &pcg))

class Tracker {
  var accessed: Bool = false
}

let rawItems: [Tracker] = Array(0..<512).map{ _ in Tracker() }
// A dataset that applies a lazy transformation on those raw items (think
// opening an image
let dataset = rawItems.lazy.map { (x: Tracker) -> Tensor<Float> in
  x.accessed = true
  return Tensor<Float>(randomNormal: [224, 224, 3])
}

final class EpochsTests: XCTestCase {
  func resetRawItems() {
    let _ = rawItems.map { $0.accessed = false }
  }
  
  func testLazyShuffle() {
    // Using `.shuffled()` access all elements
    let _ = dataset.shuffled(using: &pcg)
    XCTAssert(rawItems.allSatisfy { $0.accessed }) 

    // Using `.innerShuffled()` on a `ReindexedColletion` does not access elements
    resetRawItems()
    let _ = ReindexedCollection(dataset).innerShuffled(using: &pcg)
    XCTAssert(rawItems.allSatisfy { !$0.accessed }) 
  }
  
  func testBaseUse() {
    // `inBatches` splits our dataset in batches, the `collated` property is
    // defined for any struct conforming to `Collatable`
    let batches = dataset.inBatches(of: 64).lazy.map(\.collated)
    
    resetRawItems()
    for (i, batch) in batches.enumerated() {
      XCTAssertEqual(batch.shape, TensorShape([64, 224, 224, 3]))
      let limit = (i + 1) * 64
      XCTAssert(rawItems[..<limit].allSatisfy(\.accessed))
      XCTAssert(rawItems[limit...].allSatisfy( {!$0.accessed }))
    }
  }
    
  // Tests with shuffle
  func testShuffle() {
    let trainingEpochs = UniformTrainingEpochs(samples: dataset, batchSize: 64, 
                                               entropy: pcg)
    var accessed = Array(0..<512)
    for batches in trainingEpochs.prefix(20) {
      resetRawItems()
      var newAccessed: [Int] = []
      for batch in batches {
        let collatedBatch = batch.collated
        XCTAssertEqual(collatedBatch.shape, TensorShape([64, 224, 224, 3]))
          
        newAccessed += Array(0..<512).filter { rawItems[$0].accessed }    
      }
      XCTAssertNotEqual(accessed, newAccessed, 
                       "Dataset should have been reshuffled.")
      
      accessed = newAccessed
      let uniqueSamples = Set(accessed)
      XCTAssertEqual(
        uniqueSamples.count, rawItems.count,
        "Every epoch sample should be drawn from a different input sample.")
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

  func testNonuniformTrainingEpochs() {
    class Sample {
      init(size: Int) { self.size = size }
      var size: Int
    }

    let sampleCount = 503
    let batchSize = 7
    let samples = (0..<sampleCount).map {
      _ in Sample.init(size: Int.random(in: 0..<1000, using: &pcg))
    }
    // TODO: More thorough testing.

    let epochs = NonuniformTrainingEpochs(
      samples: samples,
      batchSize: batchSize,
      entropy: pcg) { $0.size < $1.size }

    // The first sample ordering observed during this test.
    var observedSampleOrder: [ObjectIdentifier]?

    for batches in epochs.prefix(20) {
      XCTAssertEqual(batches.count, sampleCount / batchSize)
      XCTAssert(batches.allSatisfy { $0.count == batchSize })
      let epochSamples = batches.joined()
      let epochSampleOrder = epochSamples.lazy.map(ObjectIdentifier.init)

      if let o = observedSampleOrder {
        XCTAssertFalse(
          o.elementsEqual(epochSampleOrder),
          "Batches should be randomized")
      }
      else {
        observedSampleOrder = Array(epochSampleOrder)
      }

      let maxEpochSampleSize = epochSamples.lazy.map(\.size).max()!
      XCTAssertEqual(
        batches.first!.lazy.map(\.size).max(),
        maxEpochSampleSize,
        "The first batch should contain a sample of maximal size.")

      let uniqueSamples = Set(epochSampleOrder)
      XCTAssertEqual(
        uniqueSamples.count, epochSamples.count,
        "Every epoch sample should be drawn from a different input sample.")
    }
  }
}

extension EpochsTests {
  static var allTests = [
    ("testLazyShuffle", testLazyShuffle),
    ("testBaseUse", testBaseUse),
    ("testShuffle", testShuffle),
    ("testAllPadding", testAllPadding),
    ("testSortAndPadding", testSortAndPadding),
    ("testSortishAndPadding", testSortishAndPadding),
    ("testLanguageModel", testLanguageModel),
    ("testLanguageModelShuffled", testLanguageModelShuffled),
    ("testNonuniformTrainingEpochs", testNonuniformTrainingEpochs),
  ]
}

