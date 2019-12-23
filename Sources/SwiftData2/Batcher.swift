import TensorFlow

public func identity<C>(_ x: C) -> C { return x }

public func defaultSample<C: Collection>(on dataset: C, shuffled: Bool) -> [Int] {
    return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
}

public struct Batcher<C: Collection>: Sequence where C.Index == Int {
    public let dataset: C
    public var batchSize: Int
    public var numWorkers: Int = 1
    public var shuffle: Bool = false
    public var dropLast: Bool = false
    public let sampleIndices: (C, Bool) -> [Int]
    public let padSamples: ([C.Element]) -> [C.Element]
    public let collateSamples: ([C.Element]) -> C.Element
    
    public var count: Int {
        let nSamples = dataset.count
        return nSamples / batchSize + (nSamples % batchSize == 0 || dropLast ? 0 : 1)
    }
    
    public init(on dataset: C, 
                batchSize: Int, 
                numWorkers: Int = 1, 
                shuffle: Bool = false, 
                dropLast: Bool = false,
                sampleIndices: @escaping (C, Bool) -> [Int] = defaultSample,
                padSamples: @escaping ([C.Element]) -> [C.Element] = identity,
                collateSamples: @escaping ([C.Element]) -> C.Element) {
        (self.dataset, self.batchSize,self.numWorkers, self.shuffle) = (dataset, batchSize, numWorkers, shuffle)
        (self.dropLast, self.sampleIndices, self.padSamples) = (dropLast, sampleIndices, padSamples)
        self.collateSamples = collateSamples
    }
    
    public func makeIterator() -> BatchIterator<C> {
        return BatchIterator(self)
    }
}

public struct BatchIterator<C: Collection>: IteratorProtocol where C.Index == Int{
    let b: Batcher<C>
    let indices: [Int]
    let samplesCount: Int
    var pos: Int = 0
    
    init(_ b: Batcher<C>) { 
        self.b = b
        pos = 0
        indices = b.sampleIndices(b.dataset, b.shuffle)
        samplesCount = b.dataset.count
    }
    
    public mutating func next() -> C.Element? {
        guard pos < samplesCount else { return nil }
        let end = min(pos + b.batchSize, samplesCount)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        //The idea is to have samples processed and collated on the CPU before moving to the host, not sure this is the right way.
        return withDevice(.cpu) { () -> C.Element in
            let samples = Array(pos..<end).concurrentMap(nthreads: b.numWorkers) {
                b.dataset[indices[$0]]
            }
            pos = end
            return b.collateSamples(b.padSamples(samples))
        }
    }
}

public func collateTensors<S1: TensorFlowScalar, S2: TensorFlowScalar>(
    _ samples: [(Tensor<S1>, Tensor<S2>)]
) -> (Tensor<S1>, Tensor<S2>) {
    return (Tensor(stacking: samples.map { $0.0 }), Tensor(stacking: samples.map { $0.1 }))
}