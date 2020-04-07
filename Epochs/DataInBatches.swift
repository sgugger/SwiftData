/// For type that represent a `Collection` with a special shuffled method (that
/// differs from the shuffled method provided by `Collection`).
public protocol SamplingProtocol {
  associatedtype DataSet: Collection
  /// A non-shuffled version of `Self`
  var samples: DataSet { get }
  /// A shuffeld version of `Self`
  var shuffled: DataSet { get }
}

/// A collection built lazily on an array of `Raw` items
public struct Samples<Raw, DataSet: Collection>: SamplingProtocol {
  /// The underlying raw items
  private let base: [Raw]
  /// Creates the `Dataset` form the raw items
  private let makeDataset: ([Raw]) -> DataSet
    
  public init(on base: [Raw], _ makeDataset: @escaping ([Raw]) -> DataSet) {
    self.base = base
    self.makeDataset = makeDataset
  }
    
  /// A non-shuffled version of `Self`
  public var samples: DataSet { makeDataset(base) }
  /// A shuffled version of `Self`
  public var shuffled: DataSet { makeDataset(base.shuffled()) }
}

/// An infinite generator of training and validation data in batches 
///
/// - Note: if the `batchSize` changes during one epoch, it will only be
///   reflected at the next.
public struct BatchesGenerator<BatchSamples: SamplingProtocol, Batch> {
  public typealias BatchSampleSet = BatchSamples.DataSet
  /// Training dataset
  public let trainingSamples: BatchSamples
  /// Validation dataset
  public let validationSamples: BatchSamples
  /// The batch size
  public var batchSize: Int
  /// How to make a `Batch` from a slice of `BatchSampleSet`
  private let makeBatch: (BatchSampleSet.SubSequence) -> Batch
    
  public init(
    of batchSize: Int,
    from trainingSamples: BatchSamples, 
    and validationSamples: BatchSamples,
    _ makeBatch: @escaping (BatchSampleSet.SubSequence) -> Batch
  ) {
    self.batchSize = batchSize
    self.trainingSamples = trainingSamples
    self.validationSamples = validationSamples
    self.makeBatch = makeBatch
  }
    
  /// Returns new `Batches` for training and validation, with a reshuffle of 
  /// the training data
  public func nextEpoch() -> (
    training: Batches<BatchSampleSet, Batch>, 
    validation: Batches<BatchSampleSet, Batch>
  ) {
    return (
      training: Batches(of: batchSize, from: trainingSamples.shuffled, makeBatch), 
      validation: Batches(of: batchSize, from: validationSamples.samples, makeBatch))
    }
}