import TensorFlow

public protocol BatcherConfig {
    associatedtype Item
    associatedtype Sample
    associatedtype RawBatch
    associatedtype Batch
    
    func processItem(_ item: Item) -> Sample
    func processSamples(_ samples: [Sample]) -> [Sample]
    func collate(_ samples: [Sample]) -> RawBatch 
    func processBatch(_ batch: RawBatch) -> Batch
}

public struct Batcher<C>: Sequence where C: BatcherConfig {
    public let config: C
    public let dataset: [C.Item]
    public var batchSize: Int
    public var numWorkers: Int = 1
    public var shuffle: Bool = false
    public var dropLast: Bool = false
    
    public var count: Int {
        return dataset.count / batchSize + (dataset.count % batchSize == 0 || dropLast ? 0 : 1)
    }
    
    public init(_ config: C, 
                dataset: [C.Item], 
                batchSize: Int, 
                numWorkers: Int = 1, 
                shuffle: Bool = false, 
                dropLast: Bool = false) {
        (self.config, self.dataset, self.batchSize) = (config, dataset, batchSize)
        (self.numWorkers, self.shuffle, self.dropLast) = (numWorkers, shuffle, dropLast)
    }
    
    public func makeIterator() -> BatchIterator<C> {
        return BatchIterator(self)
    }
}

public struct BatchIterator<C>: IteratorProtocol where C: BatcherConfig {
    let b: Batcher<C>
    let indices: [Int]
    var pos: Int = 0
    
    init(_ b: Batcher<C>) { 
        self.b = b
        pos = 0
        indices = b.shuffle ? Array(0..<b.dataset.count).shuffled() : Array(0..<b.dataset.count)
    }
    
    public mutating func next() -> C.Batch? {
        guard pos < b.dataset.count else { return nil }
        let end = min(pos + b.batchSize, b.dataset.count)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        //The idea is to have samples processed and collated on the CPU before moving to the host, not sure this is the right way.
        let batch = withDevice(.cpu) { () -> C.RawBatch in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.config.processItem(b.dataset[indices[$0]])
            }
            pos = end
            return b.config.collate(b.config.processSamples(samples))
        }
        return b.config.processBatch(batch)
    }
}