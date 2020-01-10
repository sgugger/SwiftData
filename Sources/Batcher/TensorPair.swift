import TensorFlow

//To avoid using a tuple of Tensor (that can't conform to anything)
public struct TensorPair<S1: TensorFlowScalar, S2: TensorFlowScalar>: Collatable, KeyPathIterable {
    public var input:  Tensor<S1>
    public var target: Tensor<S2>
    
    public init(input: Tensor<S1>, target: Tensor<S2>) {
        (self.input, self.target) = (input, target)
    }
}