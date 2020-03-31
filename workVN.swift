/// Types that represent a way to create samples from a `dataset`.
/// 
/// - Note: While the basic use case will just gather elements from `dataset`
/// using the indices in `samplesOrder`, there are some cases where the samples
/// are not direct elements of the `dataset`. For instance, in a language model
/// the samples are texts of fixed lengths built from a stream obtained by 
/// concatenating all the texts in `dataset`
public protocol SampleRuns {
    associatedtype SourceDataSet: RandomAccessCollection
    var dataset: SourceDataSet { get set }
    var samplesOrder: [Int] { get set }
    mutating func shuffle()
}

// Should we just make SampleRuns conform to Collection? 
public extension SampleRuns {
    /// Default subscript when `Self` is just a simple wrapper around `dataset`
    subscript(i: Int) -> SourceDataSet.Element {
        return dataset[dataset.index(dataset.startIndex, offsetBy: samplesOrder[i])]
    }
    
    /// Default count when `Self` is just a simple wrapper around `dataset`
    ///
    /// - Note: `samplesOrder` may contain more elements than the dataset
    /// as in oversampling strategies
    public var count: Int { return samplesOrder.count }
}

/// Base wrapper around a `dataset`
public struct DefaultSampleRuns<SourceDataSet: RandomAccessCollection>: SampleRuns {
    /// The `dataset` wrapped
    public var dataset: SourceDataSet
    /// The order to use for the samples
    public var samplesOrder: [Int]
    
    /// Creates from `dataset`, using `0..<dataset.count` as an ordering
    public init(dataset: SourceDataSet) {
        self.dataset = dataset
        samplesOrder = Array(0..<dataset.count) 
    }
    
    /// Shuffles `samplesOrder`
    public mutating func shuffle() {
        samplesOrder.shuffle()
    }
}

/// Types whose elements can be collated in some higher-rank element of the 
/// same type (example: tensors, tuple of tensors)
public protocol Collatable {
    init(collating: [Self])
}

/// Returns `x`
public func identity<T>(x: T) -> T { x }

public extension Array where Element: Collatable {
    /// Returns the element obtained by collating `self`, using `resizer` for
    /// making all elements of the same size
    func collating(with resizer: (Self) -> Self = identity) -> Element {
        return Element(collating: resizer(self))
    }
}

/// A collection of batches built on some type conforming to `SampleRuns`
public struct Batches<Samples: SampleRuns> where Samples.SourceDataSet.Element: Collatable {
    public typealias Element = Samples.SourceDataSet.Element
    /// The samples that will be assembled in batches
    private let samples: Samples
    /// The size of each batch.
    private let batchSize: Int
    /// Optionally set a limit to the number of threads used.
    private let threadsLimit: Int?
    /// If `true`, drop the last batch if it has less elements than `batchSize`.
    private let dropRemainder: Bool
    /// Function to use to make all elements the same size before collating them in a batch
    private let resizer: ([Element]) -> [Element]
    
    public init(on samples: Samples, 
                batchSize: Int,
                threadsLimit: Int? = nil,
                dropRemainder: Bool = false,
                resizer: @escaping ([Element]) -> [Element] = identity) {
        self.samples = samples
        self.batchSize = batchSize
        self.threadsLimit = threadsLimit
        self.dropRemainder = dropRemainder
        self.resizer = resizer
    }
}


extension Batches: Collection {
    public typealias Index = Int 
    public var startIndex: Int { return 0 }
    public var endIndex: Int { 
        let n = samples.count
        return n / batchSize + (n % batchSize == 0 || dropRemainder ? 0 : 1)
    }
    
    public func index(after i: Int) -> Int { i+1 }
    
    /// Access the i-th batch
    public subscript(i: Int) -> Element {
        get {
            let start = i * batchSize
            let end = Swift.min(start+batchSize, samples.count)
            let n = threadsLimit == nil ? 1 : (end - start) / threadsLimit!
            let toCollate = Array(start..<end).map {//concurrentMap(minBatchSize: n) {
                samples[$0]
            }
            return toCollate.collating(with: resizer)
        }
    }
}

public extension SampleRuns where SourceDataSet.Element: Collatable{ 
    /// Make batches from `self`, potentially shuffled
    mutating func makeBatches(
        batchSize: Int,
        shuffle: Bool = false,
        threadsLimit: Int? = nil,
        dropRemainder: Bool = false,
        resizer: @escaping ([SourceDataSet.Element]) -> [SourceDataSet.Element] = identity
    ) -> Batches<Self> {
        if shuffle { self.shuffle() }
        return Batches(
            on: self, 
            batchSize: batchSize, 
            threadsLimit: threadsLimit, 
            dropRemainder: dropRemainder,
            resizer: resizer
        )
    }
}