/// For type that represent how to build `Batches` from training or validation
/// samples.
public protocol BatchesMaker {
  associatedtype Samples: Collection
  associatedtype Batch

  func makeTraining(of batchSize: Int, from: Samples) -> Batches<Samples, Batch>
  func makeValidation(of batchSize: Int, from: Samples) -> Batches<Samples, Batch>
}

/// An infinite generator of training and validation data in batches 
///
/// - Note: if the `batchSize` changes during one epoch, it will only be
///   reflected at the next.
public struct BatchesGenerator<Maker: BatchesMaker> {
  /// Training dataset.
  public let training: Maker.Samples
  /// Validation dataset.
  public let validation: Maker.Samples
  /// The batch size.
  public var batchSize: Int
  /// How to make a `Batch` from a slice of `BatchSampleSet`.
  private let maker: Maker
  
  /// Creates an instance that will be able to generate `Batches` of `batchSize`
  /// from `training`and `validation` samples, using `maker`
  public init(
    of batchSize: Int,
    from training: Maker.Samples, 
    and validation: Maker.Samples,
    with maker: Maker
  ) {
    self.batchSize = batchSize
    self.training = training
    self.validation = validation
    self.maker = maker
  }
    
  /// Returns new `Batches` for training and validation, with a reshuffle of 
  /// the training data
  public func nextEpoch() -> (
    training: Batches<Maker.Samples, Maker.Batch>, 
    validation: Batches<Maker.Samples, Maker.Batch>
  ) {
  return (
    training: maker.makeTraining(of: batchSize, from: training), 
    validation: maker.makeValidation(of: batchSize, from: training))
  }
}

public struct LazyBatchesMaker<RawSamples: RandomAccessCollection, Batch>: BatchesMaker {
  public typealias Samples = ReindexedCollection<RawSamples>
  private let makeBatch: (Samples.SubSequence) -> Batch
 
  public init(makeBatch: @escaping (Samples.SubSequence) -> Batch) {
    self.makeBatch = makeBatch
  }
    
  public func makeTraining(of batchSize: Int, from samples: Samples) -> Batches<Samples, Batch> {
    return Batches(of: batchSize, from: samples.innerShuffled(), makeBatch)
  }
    
  public func makeValidation(of batchSize: Int, from samples: Samples) -> Batches<Samples, Batch> {
    return Batches(of: batchSize, from: samples, makeBatch)
  }
}