import Path
import TensorFlow

public struct ImageClassificationBatcher<Scalar>: Batcher where Scalar: Numeric&TensorFlowScalar {
    public typealias Item = Path
    public typealias Sample = (Tensor<Scalar>, Int)
    public typealias RawBatch = (Tensor<Scalar>, Tensor<Int32>)
    public typealias Batch = (Tensor<Float>, Tensor<Int32>)
    
    public let openImage: (Path) -> Tensor<Scalar>
    public let labelFunc: (Path) -> Int
    public let divFactor: Float
    public let dataset: [Item]
    public var batchSize: Int
    public var numWorkers: Int
    public var shuffle: Bool
    public var dropLast: Bool
    
    public init(openImage: @escaping (Path) -> Tensor<Scalar>, 
                labelFunc: @escaping (Path) -> Int, 
                divFactor: Float = 255,
                dataset: [Item],
                batchSize: Int,
                numWorkers: Int = 1,
                shuffle: Bool = false,
                dropLast: Bool = false) {
        (self.openImage, self.labelFunc, self.divFactor) = (openImage, labelFunc, divFactor)
        (self.dataset, self.batchSize, self.numWorkers) = (dataset, batchSize, numWorkers)
        (self.shuffle, self.dropLast) = (shuffle, dropLast)
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