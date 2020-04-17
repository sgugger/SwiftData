/// An infinite sequence of collections of batch samples suitable for training a
/// DNN when samples are uniform.
///
/// - Parameter `Samples`: the type of collection from which samples will be
///   drawn.
/// - Parameter `Entropy`: a source of entropy used to randomize sample order in
///   each epoch.  See the `init` documentation for details.
///
/// The batches in each epoch all have exactly the same size.
public final class UniformTrainingEpochs<
  Samples: Collection,
  Entropy: RandomNumberGenerator
> : Sequence, IteratorProtocol {
  private let samples: Samples
  
  /// The number of samples in a batch.
  let batchSize: Int

  /// The ordering of samples in the current epoch.
  private var sampleOrder: [Samples.Index]

  // TODO: Figure out how to handle non-threasafe PRNGs with a parallel shuffle
  // algorithm.
  /// A source of entropy for shuffling samples.
  private var entropy: Entropy

  /// Creates an instance drawing samples from `samples` into batches of size
  /// `batchSize`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is determinstic and not dependent on
  ///     other operations.
  public init(
    samples: Samples,
    batchSize: Int,
    entropy: Entropy
  ) {
    self.samples = samples
    self.batchSize = batchSize
    sampleOrder = Array(samples.indices)
    self.entropy = entropy
  }

  /// The type of each epoch, a collection of batches of samples.
  public typealias Element = Slices<
    LazilySelected<Samples, Array<Samples.Index>.SubSequence>>

  /// Returns the next epoch in sequence.
  public func next() -> Element? {
    let remainder = sampleOrder.count % batchSize

    // TODO: use a parallel shuffle like mergeshuffle
    // (http://ceur-ws.org/Vol-2113/paper3.pdf)
    sampleOrder.shuffle(using: &entropy)
    
    return samples.selecting(sampleOrder.dropLast(remainder))
      .inBatches(of: batchSize)
  }
}