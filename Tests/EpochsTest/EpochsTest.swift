import TensorFlow
import Epochs
import XCTest
import PcgRandom

// Seeds for reproducibility
var pcg = Pcg64Random(seed: 42)
let tfSeed: TensorFlowSeed = (
  graph: Int32.random(in:Int32.min..<Int32.max, using: &pcg), 
  op: Int32.random(in:Int32.min..<Int32.max, using: &pcg))

final class EpochsTests: XCTestCase {
  // A mock item type that tracks if it was accessed or not
  class Tracker {
    var accessed: Bool = false
  }

  let rawItems: [Tracker] = Array(0..<512).map{ _ in Tracker() }  
   
  func resetRawItems() {
    let _ = rawItems.map { $0.accessed = false }
  }
   
  // A dataset that applies a lazy transformation on those raw items (think
  // opening an image
  var dataset: LazyMapSequence<[Tracker], Tensor<Float>> { 
    return rawItems.lazy.map { (x: Tracker) -> Tensor<Float> in
      x.accessed = true
      return Tensor<Float>(randomNormal: [224, 224, 3]) }
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
    // Using `dataset.shuffled()` would break the laziness. Plus we would need
    // to do it at each new epoch. `TrainingEpochs` automatically handles
    // shuffling (and re-shuffling at each epoch) without breaking the laziness.
    let epochs = TrainingEpochs(samples: dataset, batchSize: 64, entropy: pcg)
    var accessed = Array(0..<512)
    for batches in epochs.prefix(10) {
      resetRawItems()
      var newAccessed: [Int] = []
      for batch in batches {
        XCTAssertEqual(batches.count, 8)
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
    
  // Tests with shuffle
  func testRemainderDropped() {
    // `TrainingEpochs` automatically drops the remainder batch if it has
    // less than `batchSize` elements.
    let epochs = TrainingEpochs(samples: dataset[..<500], batchSize: 64, 
                                entropy: pcg)
    let samplesCount = 500 - 500 % 64
    var accessed = Array(0..<samplesCount)
    for batches in epochs.prefix(2) {
      XCTAssertEqual(batches.count, 7)
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
        uniqueSamples.count, samplesCount,
        "Every epoch sample should be drawn from a different input sample.")
    }
  }
    
  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let nonuniformDataset: [Tensor<Int32>] = { 
    var dataset: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataset.append(Tensor<Int32>(
                       randomUniform: [Int.random(in: 1...200, using: &pcg)], 
                       lowerBound: Tensor<Int32>(0), 
                       upperBound: Tensor<Int32>(100),
                       seed: tfSeed
                    ))
    }
    return dataset
  }()
    
  func paddingTest(padValue: Int32, padFirst: Bool) {
    let batches = nonuniformDataset.inBatches(of: 64)
      .lazy.map { $0.paddedAndCollated(with: padValue) }
    for (i, b) in batches.enumerated() {
      let shapes = nonuniformDataset[(i * 64)..<((i + 1) * 64)]
        .map { Int($0.shape[0]) }
      let expectedShape = shapes.reduce(0) { max($0, $1) }
      XCTAssertEqual(Int(b.shape[1]), expectedShape)
        
      for k in 0..<64 {
        let currentShape = nonuniformDataset[i * 64 + k].shape[0]
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
    let sortedDataset = nonuniformDataset.sorted { $0.shape[0] > $1.shape[0] }
      
    let batches = sortedDataset.inBatches(of: 64)
      .lazy.map { $0.paddedAndCollated(with: 0) }
    var previousSize: Int? = nil
    for batch in batches {
      if let size = previousSize {
        XCTAssert(size >= batch.shape[1])
      }
      previousSize = Int(batch.shape[1])
    }
  }
    
  let cuts = [0, 5, 8, 15, 24, 30]
  var texts: [[Int]] { (0..<5).map { Array(cuts[$0]..<cuts[$0+1]) }}
  
  // To reindex the dataset such that the first batch samples are given by
  // indices (0, batchCount, batchCount * 2, ...
  func preBatchTranspose<C: Collection>(_ base: C, for batchSize: Int) 
  -> [C.Index] {
    let batchCount = base.count / batchSize
    return (0..<base.count).map { (i: Int) -> C.Index in 
      let j = batchCount * (i % batchSize) + i / batchSize 
      return base.index(base.startIndex, offsetBy: j) 
    }
  }
    
  //Now let's look at what it gives us:
  func testLanguageModel() {
    let sequenceLength = 3
    let batchSize = 2

    let sequences = texts.joined()
      .inBatches(of: sequenceLength)
    let indices = preBatchTranspose(sequences, for: batchSize)
    let batches = sequences.selecting(indices).inBatches(of: batchSize)
      
    var results: [[Int32]] = [[], []]
    for batch in batches { 
      let tensor = Tensor<Int32>(batch.map { Tensor<Int32>(
        $0.map { Int32($0) })})
      XCTAssertEqual(tensor.shape, TensorShape([2, 3]))
      results[0] += tensor[0].scalars
      results[1] += tensor[1].scalars
    }
    XCTAssertEqual(results[0] + results[1], (0..<30).map { Int32($0) })
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
    let sequenceLength = 3
    let batchSize = 2

    let sequences = texts.shuffled().joined()
      .inBatches(of: sequenceLength)
    let indices = preBatchTranspose(sequences, for: batchSize)
    let batches = sequences.selecting(indices).inBatches(of: batchSize)
      
    var results: [[Int32]] = [[], []]
    for batch in batches { 
      let tensor = Tensor<Int32>(batch.map { Tensor<Int32>(
        $0.map { Int32($0) })})
      XCTAssertEqual(tensor.shape, TensorShape([2, 3]))
      results[0] += tensor[0].scalars
      results[1] += tensor[1].scalars
    }
    let stream = (results[0] + results[1]).map { Int($0) }
    XCTAssertEqual(stream.count, 30)
    XCTAssert(texts.allSatisfy{ isSubset($0, from: stream) })
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
    ("testBaseUse", testBaseUse),
    ("testShuffle", testShuffle),
    ("testRemainderDropped", testRemainderDropped),
    ("testAllPadding", testAllPadding),
    ("testSortAndPadding", testSortAndPadding),
    ("testLanguageModel", testLanguageModel),
    ("testLanguageModelShuffled", testLanguageModelShuffled),
    ("testNonuniformTrainingEpochs", testNonuniformTrainingEpochs),
  ]
}

