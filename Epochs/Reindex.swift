/// A collection obtained by re-indexing a base collection.
public struct ReindexedCollection<Base: RandomAccessCollection> {
  /// The new index
  private var reindex: [Base.Index]
  /// The base collection  
  private let base: Base
  
  /// Creates an instance from `base` and `reindex`.
  public init(base: Base, reindex: [Base.Index]) {
    self.reindex = reindex
    self.base = base
  }
    
  /// Creates an instance from `base` and its indices.
  public init(_ base: Base) {
    self.reindex = Array(base.indices)
    self.base = base
  }
}

extension ReindexedCollection: RandomAccessCollection {
  public typealias Element = Base.Element
    
  /// A type whose instances represent positions in `self`.
  public typealias Index = Int

  /// The position of the first element.
  public var startIndex: Index {
    return reindex.startIndex
  }

  /// The position one past the last element.
  public var endIndex: Index {
    return reindex.endIndex
  }

  /// Returns the position immediately following `i`.
  public func index(after i: Index) -> Index {
    return reindex.index(after: i)
  }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Element {
    base[reindex[i]]
  }
}

private func sortIndexInBatches<Index>(
  of batchSize: Int, index: [Index], by areInOrder: (Index, Index) -> Bool
) -> [Index] {
  var r: [Index] = []
  r.reserveCapacity(index.count)
  var remaining = index[...]
  while !remaining.isEmpty {
    r += remaining.prefix(batchSize).sorted(by: areInOrder)
    remaining = remaining.dropFirst(batchSize)
  }
  return r
}

public extension ReindexedCollection {
  /// Shuffles its `reindex` (thus not breaking the laziness if `base` is lazy
  mutating func innerShuffle() {
    reindex.shuffle()
  }
    
  /// Returns an instance of self with `reindex` shuffled.
  func innerShuffled() -> Self {
    return ReindexedCollection(base: base, reindex: reindex.shuffled())
  }
 
  /// Sorts itself in batches
  mutating func sortInBatches(
      of batchSize: Int, by areInOrder: (Base.Index, Base.Index)->Bool
  ){
    reindex = sortIndexInBatches(of: batchSize, index: reindex, by: areInOrder)
  }
  
  func sortedInBatches(
      of batchSize: Int, by areInOrder: (Base.Index, Base.Index)->Bool
  ) -> Self {
    return ReindexedCollection(base: base, reindex: sortIndexInBatches(of: batchSize, index: reindex, by: areInOrder))
  }
}