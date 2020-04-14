import TensorFlow
import Batcher2
import XCTest

final class Batcher2Tests : XCTestCase {
  // Base use
  func test0() {
    // Some raw items (for instance filenames)
    let rawItems = 0..<512
    // A heavy-compute function lazily mapped on it (for instance, opening the images)
    let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }

    // A `Batcher` defined on this:
    var batcher = Batcher(on: dataSet, batchSize: 64)
    // Iteration over it (for instance, on epoch of the training loop) 
    batcher.reorder(shuffled: true)
    print(batcher.map(\.shape))
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
   
  // We need to pad those tensors to make them all the same length.
  // We could do this in one lazy transform applied beforehand and pad everything
  // to the same length, but it's not memory-efficient: some batches might need less
  // padding. So we need to add the padding after having selected the samples we
  // are trying to batch.
  public struct BasicPaddingTemplate<SourceDataSet: RandomAccessCollection>: BatcherTemplate 
  where SourceDataSet.Element == Tensor<Int32> {
    public func padSamples(samples: [Tensor<Int32>]) -> [Tensor<Int32>] {
      let maxLength = samples.map{ $0.shape[0] }.max()!
      return samples.map { (t: Tensor<Int32>) -> Tensor<Int32> in 
        let remaining = Tensor<Int32>(zeros: [maxLength - t.shape[0]])
        return Tensor<Int32>(concatenating: [t, remaining])
      }
    }
  }

  func test1() {
    var batcher = Batcher(with: BasicPaddingTemplate(), on: dataSet, batchSize: 64)
    batcher.reorder()
    print(batcher.map(\.shape))
  }

  // Use with a sampler
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  public struct AdvancedPaddingTemplate<SourceDataSet: RandomAccessCollection>: BatcherTemplate 
  where SourceDataSet.Element == Tensor<Int32> {
    public func padSamples(samples: [Tensor<Int32>]) -> [Tensor<Int32>] {
      let maxLength = samples.map{ $0.shape[0] }.max()!
      return samples.map { (t: Tensor<Int32>) -> Tensor<Int32> in 
        let remaining = Tensor<Int32>(zeros: [maxLength - t.shape[0]])
        return Tensor<Int32>(concatenating: [t, remaining])
      }
    }
    
    public func sampleIndices(on dataset: [Tensor<Int32>]) -> [Int] {
      return Array(0..<dataset.count).sorted { dataset[$0].shape[0] > dataset[$1].shape[0] }
    }
  }

  func test2() {
    // Use with a sampler
    // In our previous example, another way to be memory efficient is to batch
    // samples of roughly the same lengths.
    var batcher = Batcher<AdvancedPaddingTemplate<[Tensor<Int32>]>, [Tensor<Int32>]>(
      with: AdvancedPaddingTemplate(), on: dataSet, batchSize: 64)
    batcher.reorder()
    print(batcher.map(\.shape))
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
  let dataset: DataSet = {
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    return DataSet(numbers: numbers, sequenceLength: 3)
  }()

  public struct ShufflingTemplate: BatcherTemplate {
    public typealias SourceDataSet = DataSet 
    public func shuffleIndices(on dataset: inout DataSet, indices: [Int]) -> [Int] {
      dataset.shuffle()
      return indices
    }
  }

  //Now let's look at what it gives us:
  func test3() {
    var batcher = Batcher(with: ShufflingTemplate(), on: dataset, batchSize: 3)
    batcher.reorder()
    print(batcher.map(\.shape))
          
    batcher.reorder(shuffled: true)
    print(batcher.map(\.shape))
  }
}


extension Batcher2Tests {
  static var allTests = [
    ("test0", test0),
    ("test1", test1),
    ("test2", test2),
    ("test3", test3),
  ]
}