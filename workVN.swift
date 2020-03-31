public protocol SampleRuns {
    associatedtype SourceDataSet: RandomAccessCollection
    var dataset: SourceDataSet { get set }
    var samplesOrder: [Int] { get set }
    mutating func shuffle()
}

public extension SampleRuns {
    subscript(i: Int) -> SourceDataSet.Element {
        return dataset[dataset.index(dataset.startIndex, offsetBy: samplesOrder[i])]
    }
    
    public var count: Int { return dataset.count }
}

public struct DefaultSampleRuns<SourceDataSet: RandomAccessCollection>: SampleRuns {
    public var dataset: SourceDataSet
    public var samplesOrder: [Int]
    
    public init(dataset: SourceDataSet) {
        self.dataset = dataset
        samplesOrder = Array(0..<dataset.count) 
    }
    
    public mutating func shuffle() {
        samplesOrder.shuffle()
    }
}

public protocol Collatable {
    init(collating: [Self])
}

public func identity<T>(x: T) -> T { x }

public extension Array where Element: Collatable {
    func collating(with resizer: (Self) -> Self = identity) -> Element {
        return Element(collating: resizer(self))
    }
}

public struct Batches<Samples: SampleRuns> where Samples.SourceDataSet.Element: Collatable {
    public typealias Element = Samples.SourceDataSet.Element
    private let samples: Samples
    private let batchSize: Int
    private let threadsLimit: Int?
    private let dropRemainder: Bool
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