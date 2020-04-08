import TensorFlow // for paddedAndCollated

/// A collection of `Batch` whose elements are successive chunks of
/// a sample set, lazily mapped through a batch creation closure.
///
/// In typical use, `Batch` will consist of one (for inference/unsupervised
/// training) or two (for supervised or semi-supervised training) tensors, and 
/// the closure stacks the chunk of samples into a `Batch`, if necessary 
/// applying a transformation first to ensure they are of the same sizes.
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
  /// - Note: if `batchSize % samples.count != 0`, the final chunk passed to
  ///   `transform` will have only `samples.count % batchSize` samples.
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

  /// Returns the elements of `self`, padded to maximal shape with `padValue`
  /// and collated.
  public func paddedAndCollated<Scalar : Numeric>(
      with padValue: Scalar, padFirst: Bool = false
  ) -> Element
  where Element == Tensor<Scalar>
  {
    let firstShape = self.first!.shapeTensor
    let otherShapes = self.dropFirst().lazy.map(\.shapeTensor)
    let paddedShape
        = otherShapes.reduce(firstShape) { TensorFlow.max($0, $1) }
        .scalars.lazy.map { Int($0) }

    let r = self.lazy.map { t in
      t.padded(
        forSizes: zip(t.shape, paddedShape).map {
          return (before: padFirst ? $1 - $0 : 0, after: padFirst ? 0 : $1 - $0)},
        with: padValue)
    }
    return r.collated
  }
}

// A weak substitute for a corresponding property on `Collection`, which we
// can't define due to language limitations or the fact that we don't have a
// tensor protocol (take your pick).
/// Returns x.collatedAndPadded(with: 0).
public func tailPaddedWith0AndCollated<C: Collection, S: Numeric>(_ x: C) -> Tensor<S>
    where C.Element == Tensor<S>
{
  return x.paddedAndCollated(with: 0)
}

public extension Collection {
  func sortedInBatches(
      of batchSize: Int, by areInOrder: (Element, Element)->Bool
  ) -> [Element] {
    var r: [Element] = []
    r.reserveCapacity(self.count)
    var remaining = self[...]
    while !remaining.isEmpty {
      r += remaining.prefix(batchSize).sorted(by: areInOrder)
      remaining = remaining.dropFirst(batchSize)
    }
    return r

    //order.inParallelOverSlices(of: batchSize ?? self.count) {
    //  batch: inout [Base.Index].SubSequence in
    //  sort(batch) { areInOrder(base[$0], base[$1]) }
    //}
  }
}
