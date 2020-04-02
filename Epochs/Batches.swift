/// A collection of `Batch` whose elements are successive chunks of
/// a sample set, lazily mapped through a batch creation closure.
///
/// In typical use, `Batch` will consist of one (for inference/unsupervised
/// training) or two (for supervised training) tensors, and the closure stacks
/// the chunk of samples into a `Batch`, if necessary applying a transformation
/// first to ensure they are uniform.
public struct Batches<BatchSampleSet: Collection, Batch> {
  /// The underlying samples.
  private let samples: BatchSampleSet
  /// The size of every batch, except possibly the last one.
  private let batchSize: Int
  /// Returns a batch given the constituent batch samples.
  private let makeBatch: (BatchSampleSet.SubSequence) -> Batch

  /// Creates an instance that presents successive chunks, of size `batchSize`,
  /// from `samples`, lazily transformed by `transform`.
  ///
  /// If `batchSize % samples.count != 0`, the final chunk presented to
  /// `transform` will have fewer samples: `(samples.count - 1) % batchSize +
  /// 1`.
  public init(
      of batchSize: Int, from samples: BatchSampleSet, 
      _ transform: @escaping (BatchSampleSet.SubSequence) -> Batch
  ) {
    self.samples = samples
    self.batchSize = batchSize
    self.makeBatch = transform
  }
}

extension Batches : Collection {
  /// A type whose instances represent positions in `self`.
  public typealias Index = BatchSampleSet.Index

  /// The position of the first element.
  public var startIndex: BatchSampleSet.Index {
    return samples.startIndex
  }

  /// The position one past the last element.
  public var endIndex: BatchSampleSet.Index {
    return samples.endIndex
  }


  /// Returns the position immediately following `i`.
  public func index(after i: Index) -> Index {
    return samples.index(i, offsetBy: batchSize, limitedBy: endIndex)!
  }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Batch {
    makeBatch(samples[i..<index(after: i)])
  }
}

extension Collection where Element: Collatable {
  /// The result of collating the elements of `self`.
  public var collated: Element { .init(collating: self) }
}

public struct SelectedElements<Base: Collection, Order: Collection>
    where Order.Element == Base.Index
{
  let base: Base
  let order: Order
  
  init(_ base: Base, order: Order) {
    self.base = base
    self.order = order
  }
}

extension SelectedElements : Collection {
  public typealias Index = Order.Index
  public typealias Element = Base.Element
  
  public var startIndex: Index { order.startIndex }
  
  public var endIndex: Index { order.endIndex }
  
  public subscript(i: Index) -> Base.Element {
    base[order[i]]
  }

  public func index(after i: Index) -> Index { order.index(after: i) }
}

public extension Collection {
  func sortedInBatches(
      batchSize: Int? = nil, by areInOrder: (Element, Element)->Bool
  ) -> SelectedElements<Self, [Index]> {
    var order = Array(self.indices)
    
    order.sort { areInOrder(self[$0], self[$1]) }
    //order.inParallelOverSlices(of: batchSize ?? self.count) {
    //  batch: inout [Base.Index].SubSequence in
    //  sort(batch) { areInOrder(base[$0], base[$1]) }
    //}
    
    return SelectedElements(self, order: order)
  }
}

// x.dropLast(count % batchSize)

