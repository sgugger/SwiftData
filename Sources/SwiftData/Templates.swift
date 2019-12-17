import Path
import TensorFlow

public struct ImageClassificationTemplate<Scalar>: BatcherTemplate where Scalar: Numeric&TensorFlowScalar {
    public typealias Item = Path
    public typealias Sample = (Tensor<Scalar>, Int)
    public typealias RawBatch = (Tensor<Scalar>, Tensor<Int32>)
    public typealias Batch = (Tensor<Float>, Tensor<Int32>)
    
    let openImage: (Path) -> Tensor<Scalar>
    let labelFunc: (Path) -> Int
    let divFactor: Float
    
    public init(openImage: @escaping (Path) -> Tensor<Scalar>, 
         labelFunc: @escaping (Path) -> Int, 
         divFactor: Float = 255) {
        (self.openImage, self.labelFunc, self.divFactor) = (openImage, labelFunc, divFactor)
    }
    
    public func processItem(_ item: Item) -> Sample {
        return (openImage(item), labelFunc(item))
    }
    
    public func collateSamples(_ samples: [Sample]) -> RawBatch {
        return (Tensor(stacking: samples.map { $0.0 }), Tensor(samples.map { Int32($0.1) }))
    }
    
    public func processBatch(_ batch: RawBatch) -> Batch {
        return (Tensor<Float>(batch.0) / self.divFactor, batch.1)
    }
}

public struct TextClassificationTemplate<Item>: BatcherTemplate {
    public typealias Sample = ([Int], Int)
    public typealias RawBatch = (Tensor<Int32>, Tensor<Int32>)
    public typealias Batch = (Tensor<Int32>, Tensor<Int32>)
    
    let openItem: (Item) -> [Int]
    let labelFunc: (Item) -> Int
    let padIndex: Int
    let padFirst: Bool
    
    public init(openItem: @escaping (Item) -> [Int], 
         labelFunc: @escaping (Item) -> Int, 
         padIndex: Int = 1,
         padFirst: Bool = false) {
        (self.openItem, self.labelFunc, self.padIndex, self.padFirst) = (openItem, labelFunc, padIndex, padFirst)
    }
    
    public func processItem(_ item: Item) -> Sample {
        return (openItem(item), labelFunc(item))
    }
    
    private func padOne(_ x: [Int], to length: Int) -> [Int] {
        let pad: [Int] = Array(repeating: padIndex, count: length - x.count)
        return padFirst ? pad + x : x + pad
    }
    
    public func processSamples(_ samples: [Sample]) -> [Sample] { 
        let maxLength = samples.map { $0.0.count }.max()!
        return samples.map { (padOne($0.0, to: maxLength), $0.1) }
    }
    
    public func collateSamples(_ samples: [Sample]) -> RawBatch {
        return (Tensor(stacking: samples.map { Tensor<Int32>($0.0.map { Int32($0) })}), 
                Tensor(samples.map { Int32($0.1) }))
    }
}

public struct LanguageModelTemplate<Item>: BatcherTemplate {
    public typealias Sample   = [Int]
    public typealias Batch    = (Tensor<Int32>, Tensor<Int32>)
    public typealias RawBatch = (Tensor<Int32>, Tensor<Int32>)
    public let openItem: (Item) -> Sample
    
    public var batchSize: Int
    public var sequenceLength: Int
    public var lengths: [Int]
    private var batchLength: Int
    private var batchCount: Int
    private var lastLength: Int
    
    public init(openItem: @escaping (Item) -> Sample,
                batchSize: Int,
                sequenceLength: Int,
                lengths: [Int]) {
        (self.openItem, self.batchSize, self.sequenceLength) = (openItem, batchSize, sequenceLength)
        self.lengths = lengths
        batchLength = (lengths.reduce(0, +) - 1) / batchSize
        batchCount = batchLength / sequenceLength + (batchLength % sequenceLength == 0 ? 0 : 1)
        lastLength = batchLength - (batchCount - 1) * sequenceLength
    }
    
    public func samplesCount(of dataset: [Item]) -> Int {
        return batchCount * batchSize
    }
    
    public func processItem(_ item: Item) -> Sample {
        return openItem(item)
    }
    
    private func readItems(from start: Int, to end: Int, in dataset: [Item], with indices: [Int]) -> [Int] {
        var res: [Int] = []
        var index = 0
        var cumLen = 0
        while cumLen + lengths[indices[index]] < start {
            cumLen += lengths[indices[index]]
            index += 1
        }
        var pos = start
        while pos < end {
            let x = processItem(dataset[indices[index]])
            let readFrom = pos - cumLen
            let readUntil = min(end - cumLen, x.count)
            res = res + Array(x[readFrom..<readUntil])
            pos = readUntil + cumLen
            cumLen += lengths[indices[index]]
            index += 1
        }
        return res
    }
    
    public func createSample(in dataset: [Item], with indices: [Int], at index: Int) -> Sample { 
        let sampleLength = index / batchSize == batchCount - 1 ? lastLength : sequenceLength
        let startIndex = (index % batchSize) * batchLength + (index / batchSize) * sequenceLength
        return readItems(from: startIndex, to: startIndex + sampleLength + 1, in: dataset, with: indices)
    }
    
    public func collateSamples(_ samples: [Sample]) -> RawBatch {
        let samples32 = samples.map { $0.map { Int32($0) } }
        return (Tensor(stacking: (samples32.map { Tensor<Int32>($0.prefix(upTo: $0.count-1)) })), 
                Tensor(stacking: (samples32.map { Tensor<Int32>($0.suffix(from: 1)) })))
    }
}
