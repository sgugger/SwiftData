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