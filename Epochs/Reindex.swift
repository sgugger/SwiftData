/// A collection obtained by re-indexing a base collection.
public struct ReindexedCollection<Reindex: Collection, Base: Collection>
where Reindex.Element == Base.Index {
  /// The new index
  private let reindex: Reindex
  /// The base collection  
  private let base: Base
  
  /// Creates an instance from `base` and `reindex`
  public init(reindex: Reindex, base: Base) {
    self.reindex = reindex
    self.base = base
  }
}

extension ReindexedCollection: Collection {
  public typealias Element = Base.Element
    
  /// A type whose instances represent positions in `self`.
  public typealias Index = Reindex.Index

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

public extension Collection {
  /// Shuffles the `indices`, thus not breaking a lazy collection
  func innerShuffled() -> ReindexedCollection<[Index], Self> {
    return ReindexedCollection(reindex: indices.shuffled(), base: self)
  }
}