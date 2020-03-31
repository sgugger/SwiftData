struct Batches<Batch, BatchSampleSet: Collection> {
  /// The size of every batch, except possibly the last one.
  let batchSize: Int
  /// Returns a batch given the constituent batch samples.
  let makeBatch: (BatchSampleSet.SubSequence) -> Batch
  /// The underlying samples.
  let samples: BatchSampleSet
}

extension Batches : Collection {
  typealias Index = BatchSampleSet.Index
  
  var startIndex: BatchSampleSet.Index {
    return samples.startIndex
  }

  var endIndex: BatchSampleSet.Index {
    return samples.endIndex
  }

  func index(after i: Index) -> Index {
    return samples.index(i, offsetBy: batchSize, limitedBy: endIndex)
  }

  subscript(i: Index) -> Batch {
    makeBatch(samples[i..<index(after: i)])
  }
}

func defaultMakeBatch<BatchSamples: Collection>(samples: BatchSamples)
    -> BatchSamples.Element
    where BatchSamples.Element: Collatable
{
  return .init(collating: samples)
}


struct BatchSorted<Base: Collection> {
  let base: Base
  let batchSize: Int
  let order: [Base.Index]
  
  init(
      _ base: Base, batchSize: Int? = nil,
      by areInOrder: (Base.Element, Base.Element)->Bool
  ) {
    self.base = base
    self.batchSize = batchSize ?? base.count
    var order = Array(base.indices)
    
    order.inParallelOverSlices(of: batchSize) {
      batch: inout [Base.Index].SubSequence in
      sort(batch) { areInOrder(base[$0], base[$1]) }
    }
  }
}

extension BatchSorted : Collection {
  var startIndex: Int { order.startIndex }
  var endIndex: Int { order.endIndex }
  subscript(i: Int) -> Base.Element {
    
  }
}

//-------


struct SelectedElements<Base: Collection, Order: Collection>
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
  typealias Index = Order.Index
  typealias Element = Base.Element
  
  var startIndex: Index { order.startIndex }
  
  var endIndex: Index { order.endIndex }
  
  subscript(i: Index) -> Base.Element {
    base[order[i]]
  }

  func index(after i: Index) -> Index { order.index(after: i) }
}

extension Collection {
  func sortedInBatches(
      batchSize: Int? = nil, by: by areInOrder: (Element, Element)->Bool
  ) -> SelectedElements<Self, [Index]> {
    var order = Array(self.indices)
    
    order.inParallelOverSlices(of: batchSize ?? self.count) {
      batch: inout [Base.Index].SubSequence in
      sort(batch) { areInOrder(base[$0], base[$1]) }
    }
    
    return SelectedElements(self, order: order)
  }
}

// x.dropLast(count % batchSize)

