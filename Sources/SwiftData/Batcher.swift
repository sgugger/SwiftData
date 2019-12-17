import TensorFlow

public protocol BatcherTemplate {
    associatedtype Item
    associatedtype Sample
    associatedtype RawBatch
    associatedtype Batch
    
    func sampleIndices(on dataset: [Item], shuffled: Bool) -> [Int]
    func samplesCount(of dataset: [Item]) -> Int
    func processItem(_ item:    Item) -> Sample
    func createSample(in dataset: [Item], with indices: [Int], at index: Int) -> Sample
    func processSamples(_ samples: [Sample]) -> [Sample]
    func collateSamples(_ samples: [Sample]) -> RawBatch 
    func processBatch (_ batch:   RawBatch) -> Batch
}

public struct Batcher<T>: Sequence where T: BatcherTemplate {
    public let template: T
    public let dataset: [T.Item]
    public var batchSize: Int
    public var numWorkers: Int = 1
    public var shuffle: Bool = false
    public var dropLast: Bool = false
    
    public var count: Int {
        let nSamples = template.samplesCount(of: dataset)
        return nSamples / batchSize + (nSamples % batchSize == 0 || dropLast ? 0 : 1)
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
    let samplesCount: Int
    var pos: Int = 0
    
    init(_ b: Batcher<T>) { 
        self.b = b
        pos = 0
        indices = b.template.sampleIndices(on: b.dataset, shuffled: b.shuffle)
        samplesCount = b.template.samplesCount(of: b.dataset)
    }
    
    public mutating func next() -> T.Batch? {
        guard pos < samplesCount else { return nil }
        let end = min(pos + b.batchSize, samplesCount)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        //The idea is to have samples processed and collated on the CPU before moving to the host, not sure this is the right way.
        let batch = withDevice(.cpu) { () -> T.RawBatch in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.template.createSample(in: b.dataset, with: indices, at: $0)
            }
            pos = end
            return b.template.collateSamples(b.template.processSamples(samples))
        }
        return b.template.processBatch(batch)
    }
}

public extension BatcherTemplate {
    func sampleIndices(on dataset: [Item], shuffled: Bool) -> [Int] {
        return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
    }
    
    func samplesCount(of dataset: [Item]) -> Int { return dataset.count }
    
    func processSamples(_ samples: [Sample]) -> [Sample] { return samples }
    
    func createSample(in dataset: [Item], with indices: [Int], at index: Int) -> Sample {
        return processItem(dataset[indices[index]])
    }
}

public extension BatcherTemplate where Item == Sample {
    func processItem(_ item: Item) -> Sample { return item }
}


public extension BatcherTemplate where RawBatch == Batch {
    func processBatch(_ batch: RawBatch) -> Batch { return batch }
}