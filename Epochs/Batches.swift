public struct Batches<Batch, BatchSampleSet: Collection> {
  /// The underlying samples.
  let samples: BatchSampleSet
  /// The size of every batch, except possibly the last one.
  let batchSize: Int
  /// Returns a batch given the constituent batch samples.
  let makeBatch: (BatchSampleSet.SubSequence) -> Batch
  
  public init(on samples: BatchSampleSet, batchSize: Int, 
              makeBatch: @escaping (BatchSampleSet.SubSequence) -> Batch) {
    self.samples = samples
    self.batchSize = batchSize
    self.makeBatch = makeBatch
  }
}

extension Batches : Collection {
  public typealias Index = BatchSampleSet.Index
  
  public var startIndex: BatchSampleSet.Index {
    return samples.startIndex
  }

  public var endIndex: BatchSampleSet.Index {
    return samples.endIndex
  }

  public func index(after i: Index) -> Index {
    return samples.index(i, offsetBy: batchSize, limitedBy: endIndex)!
  }

  public subscript(i: Index) -> Batch {
    makeBatch(samples[i..<index(after: i)])
  }
}

public func defaultMakeBatch<BatchSamples: Collection>(samples: BatchSamples)
    -> BatchSamples.Element
    where BatchSamples.Element: Collatable
{
  return .init(collating: samples)
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

