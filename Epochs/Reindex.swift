/// A collection obtained by re-indexing a base collection.
public struct ReindexedCollection<Base: RandomAccessCollection> {
  /// The new index
  private var reindex: [Base.Index]!
  /// The base collection  
  private let base: Base
  
  /// Creates an instance from `base` and `reindex`.
  public init(_ base: Base, reindex: [Base.Index]! = nil) {
    self.reindex = reindex
    self.base = base
  }
}

extension ReindexedCollection: RandomAccessCollection {
  public typealias Element = Base.Element
    
  /// A type whose instances represent positions in `self`.
  public typealias Index = Int

  /// The position of the first element.
  public var startIndex: Int { 0 }

  /// The position one past the last element.
  public var endIndex: Int { base.count  }

  /// Returns the position immediately following `i`.
  public func index(after i: Int) -> Index { i+1 }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Element {
    if reindex==nil {
      return base[base.index(base.startIndex, offsetBy: i)]
    } else {
      return base[reindex[i]]
    }
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
    if reindex == nil { reindex = Array(base.indices) }
    reindex.shuffle()
  }
    
  /// Returns an instance of self with `reindex` shuffled.
  func innerShuffled() -> Self {
    return ReindexedCollection(
      base, reindex: (reindex ?? Array(base.indices)).shuffled())
  }
  
  /// Returns an instance of self with `reindex` shuffled with `rng`.
  func innerShuffled<T>(using generator: inout T) -> Self where T : RandomNumberGenerator {
    return ReindexedCollection(
      base, reindex: (reindex ?? Array(base.indices)).shuffled(using: &generator))
  }
 
  /// Sorts itself in batches
  mutating func sortInBatches(
      of batchSize: Int, by areInOrder: (Base.Index, Base.Index)->Bool
  ){
    reindex = sortIndexInBatches(
      of: batchSize, index: reindex ?? Array(base.indices), by: areInOrder)
  }
  
  func sortedInBatches(
      of batchSize: Int, by areInOrder: (Base.Index, Base.Index)->Bool
  ) -> Self {
    return ReindexedCollection(
      base, reindex: sortIndexInBatches(
        of: batchSize, index: reindex ?? Array(base.indices), by: areInOrder))
  }
}