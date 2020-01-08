import TensorFlow

//To avoid using a tuple of Tensor (that can't conform to anything)
public struct TensorPair<S1: TensorFlowScalar, S2: TensorFlowScalar>: TensorGroup {
    public var input:  Tensor<S1>
    public var target: Tensor<S2>
    
    public init(input: Tensor<S1>, target: Tensor<S2>) {
        (self.input, self.target) = (input, target)
    }

    //For lazy tensor
    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 2)
        let inputIndex  = _handles.startIndex
        let targetIndex = _handles.index(inputIndex, offsetBy: 1)
        input  = Tensor<S1>(handle: TensorHandle<S1>(handle: _handles[inputIndex]))
        target = Tensor<S2>(handle: TensorHandle<S2>(handle: _handles[targetIndex]))
    }
}