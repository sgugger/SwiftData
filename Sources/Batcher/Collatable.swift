import TensorFlow

//Protocol to conform to to use the default collateSamples function
public protocol Collatable {
    init(collating: [Self])
}

//Tensor are collated using stacking
extension Tensor: Collatable {
    public init(collating: [Self]) { self.init(stacking: collating) }
}

//TensorPair are collated by collating the underlying tensors
extension TensorPair: Collatable {
    public init(collating: [Self]) { 
        self.init(input:  Tensor<S1>(stacking: collating.map { $0.input }),
                  target: Tensor<S2>(stacking: collating.map { $0.target }))
    }
}