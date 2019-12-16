import TensorFlow

public protocol Batcher: Sequence {
    associatedtype Item
    associatedtype Sample
    associatedtype RawBatch
    associatedtype Batch
    
    var dataset: [Item] { get }
    var batchSize: Int {get set}
    var numWorkers: Int {get set}
    var shuffle: Bool {get set}
    var dropLast: Bool {get set}
    
    func sampleIndices (_ dataset: [Item], shuffled: Bool) -> [Int]
    func samplesCount(_ dataset: [Item]) -> Int
    func processItem(_ item:    Item) -> Sample
    func createSample(_ dataset: [Item], _ index: Int) -> Sample
    func processSamples(_ samples: [Sample]) -> [Sample]
    func collateSamples(_ samples: [Sample]) -> RawBatch 
    func processBatch (_ batch:   RawBatch) -> Batch
}

public extension Batcher {
    var count: Int {
        let nSamples = samplesCount(dataset)
        return nSamples / batchSize + (nSamples % batchSize == 0 || dropLast ? 0 : 1)
    }
}

public extension Batcher {
    func makeIterator() -> BatchIterator<Self> {
        return BatchIterator(self)
    }
}

public struct BatchIterator<B>: IteratorProtocol where B: Batcher {
    let b: B
    let indices: [Int]
    var pos: Int = 0
    
    init(_ b: B) { 
        self.b = b
        pos = 0
        indices = b.sampleIndices(b.dataset, shuffled: b.shuffle)
    }
    
    public mutating func next() -> B.Batch? {
        guard pos < b.dataset.count else { return nil }
        let end = min(pos + b.batchSize, b.dataset.count)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        //The idea is to have samples processed and collated on the CPU before moving to the host, not sure this is the right way.
        let batch = withDevice(.cpu) { () -> B.RawBatch in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.createSample(b.dataset, indices[$0])
            }
            pos = end
            return b.collateSamples(b.processSamples(samples))
        }
        return b.processBatch(batch)
    }
}

public extension Batcher {
    func sampleIndices (_ dataset: [Item], shuffled: Bool) -> [Int] {
        return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
    }
    
    func samplesCount(_ dataset: [Item]) -> Int { return dataset.count }
    
    func processSamples(_ samples: [Sample]) -> [Sample] { return samples }
    
    func createSample(_ dataset: [Item], _ index: Int) -> Sample { return processItem(dataset[index]) }
}

public extension Batcher where Item == Sample {
    func processItem(_ item: Item) -> Sample { return item }
}


public extension Batcher where RawBatch == Batch {
    func processBatch(_ batch: RawBatch) -> Batch { return batch }
}