import TensorFlow
import Epochs
import XCTest

final class EpochsTests : XCTestCase {
  // Some raw items (for instance filenames)
  let rawItems: [Int] = Array(0..<512)
  
  // Base use  
  func test0() {
   // A heavy-compute function lazily mapped on the raw items (for instance, opening the images)
    let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }

    // A `Batcher` defined on this:
    let batches = Batches(of: 64, from: dataSet, \.collated)
    // Iteration over it (for instance, on epoch of the training loop) 
    print(batches.map(\.shape))
  }
    
  // Test with shuffle
  func test1() {
    // We need to actually go back to the raw collection to shuffle:
    let dataSet = rawItems.shuffled().lazy.map { _ -> Tensor<Float> in
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    let batches = Batches(of: 64, from: dataSet, \.collated)
    print(batches.map(\.shape))
  }
    
  func test2() {
    // ReindexCollection does that for us
    let dataSet = rawItems.lazy.map { _ -> Tensor<Float> in
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    let batches = Batches(of: 64, from: ReindexedCollection(dataSet).innerShuffled(), \.collated)
    print(batches.map(\.shape))
  }
    
  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let dataSet: [Tensor<Int32>] = { 
    var dataSet: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataSet.append(Tensor<Int32>(
                       randomUniform: [Int.random(in: 1...200)], 
                       lowerBound: Tensor<Int32>(0), 
                       upperBound: Tensor<Int32>(100)
                    ))
    }
    return dataSet
  }()

  func test3() {
    let batches = Batches(of: 64, from: dataSet) { $0.paddedAndCollated(with: 0) }
    print(batches.map(\.shape))
  }

  // Use with a sampler
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  func sortSamples(on dataset: inout [Tensor<Int32>], shuffled: Bool) -> [Int] {
    // Just giving a quick example, but the function should shuffle a bit the sorted thing
    // if shuffle=true
    return Array(0..<dataset.count).sorted { dataset[$0].shape[0] > dataset[$1].shape[0] }
  }

  func test4() {
    // Use with a sampler
    // In our previous example, another way to be memory efficient is to batch
   // samples of roughly the same lengths.
    let sortedDataset = dataSet.sorted { $0.shape[0] > $1.shape[0] }
      
    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    print(batches.map(\.shape))
  }
    
  func test5() {
    // When using a `batchSize` we get a bit of shuffle:
    // This can all be applied on a lazy collection without breaking the lasziness as long as the sort function does not access the dataset
    let sortedDataset = ReindexedCollection(dataSet).innerShuffled().sortedInBatches(of: 256) { 
      dataSet[$0].shape[0] > dataSet[$1].shape[0] 
    }

    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    print(batches.map(\.shape))
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
  func test6() {
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers, sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    print(batches.map(\.shape))
  }

  func test7() {
    // To shuffle it we need to go back to the inner numbers
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers.shuffled(), sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    print(batches.map(\.shape))
  }
}


extension EpochsTests {
  static var allTests = [
    ("test0", test0),
    ("test1", test1),
    ("test2", test2),
    ("test3", test3),
    ("test4", test4),
    ("test5", test5),
    ("test6", test6),
    ("test7", test7),
  ]
}