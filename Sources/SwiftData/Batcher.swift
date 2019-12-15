import TensorFlow

public protocol BatcherTemplate {
    associatedtype Item
    associatedtype Sample
    associatedtype RawBatch
    associatedtype Batch
    
    func sampleIndices (_ dataset: [Item], shuffled: Bool) -> [Int]
    func processItem   (_ item:    Item)     -> Sample
    func processSamples(_ samples: [Sample]) -> [Sample]
    func collateSamples(_ samples: [Sample]) -> RawBatch 
    func processBatch  (_ batch:   RawBatch) -> Batch
}

public struct Batcher<T>: Sequence where T: BatcherTemplate {
    public let template: T
    public let dataset: [T.Item]
    public var batchSize: Int
    public var numWorkers: Int = 1
    public var shuffle: Bool = false
    public var dropLast: Bool = false
    
    public var count: Int {
        return dataset.count / batchSize + (dataset.count % batchSize == 0 || dropLast ? 0 : 1)
    }
    
    public init(from template: T, 
                on dataset: [T.Item], 
                batchSize: Int, 
                numWorkers: Int = 1, 
                shuffle: Bool = false, 
                dropLast: Bool = false) {
        (self.template, self.dataset, self.batchSize) = (template, dataset, batchSize)
        (self.numWorkers, self.shuffle, self.dropLast) = (numWorkers, shuffle, dropLast)
    }
    
    public func makeIterator() -> BatchIterator<T> {
        return BatchIterator(self)
    }
}

public struct BatchIterator<T>: IteratorProtocol where T: BatcherTemplate {
    let b: Batcher<T>
    let indices: [Int]
    var pos: Int = 0
    
    init(_ b: Batcher<T>) { 
        self.b = b
        pos = 0
        indices = b.template.sampleIndices(b.dataset, shuffled: b.shuffle)
    }
    
    public mutating func next() -> T.Batch? {
        guard pos < b.dataset.count else { return nil }
        let end = min(pos + b.batchSize, b.dataset.count)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        //The idea is to have samples processed and collated on the CPU before moving to the host, not sure this is the right way.
        let batch = withDevice(.cpu) { () -> T.RawBatch in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.template.processItem(b.dataset[indices[$0]])
            }
            pos = end
            return b.template.collateSamples(b.template.processSamples(samples))
        }
        return b.template.processBatch(batch)
    }
}

public extension BatcherTemplate where Item == Sample {
    func processItem(_ item: Item) -> Sample { return item }
}

public extension BatcherTemplate {
    func sampleIndices (_ dataset: [Item], shuffled: Bool) -> [Int] {
        return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
    }
    
    func processSamples(_ samples: [Sample]) -> [Sample] { return samples }
}

public extension BatcherTemplate where RawBatch == Batch {
    func processBatch(_ batch: RawBatch) -> Batch { return batch }
}