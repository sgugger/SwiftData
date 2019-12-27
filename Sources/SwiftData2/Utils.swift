import Foundation
import Path
import TensorFlow

//ThreadSafe and concurrentMap based on https://talk.objc.io/episodes/S01E90-concurrent-map
//TODO: build a proper separate module that does the parallel processing
public final class ThreadSafe<A> {
    var _value: A
    let queue = DispatchQueue(label: "ThreadSafe")
    init(_ value: A) { self._value = value }
  
    var value: A {
        return queue.sync { _value }
    }
    func atomically(_ transform: (inout A) -> ()) {
        queue.sync { transform(&self._value) }
    }
}

public extension Array {
    func concurrentMap<B>(nthreads:Int?=nil, _ transform: (Element) -> B) -> [B] {
        let result = ThreadSafe(Array<B?>(repeating: nil, count: count))
        let nt = nthreads ?? count
        let cs = (count-1)/nt+1
        DispatchQueue.concurrentPerform(iterations: nt) { i in
            let min = i*cs
            let max = min+cs>count ? count : min+cs
            for idx in (min..<max) {
                let element = self[idx]
                let transformed = transform(element)
                result.atomically { $0[idx] = transformed }
            }
        }
        return result.value.map { $0! }
    }
}

//Collect all files in path, filtering extensions, possibly looking in all subfolders.
public func collectFiles(under path: Path, recurse: Bool = false, filtering extensions: [String]? = nil) -> [Path] {
    var res: [Path] = []
    for p in try! path.ls(){
        if p.kind == .directory && recurse { 
            res += collectFiles(under: p.path, recurse: recurse, filtering: extensions)
        } else if extensions == nil || extensions!.contains(p.path.extension.lowercased()) {
            res.append(p.path)
        }
    }
    return res
}

//Operator for function composition
precedencegroup CompositionPrecedence { associativity: left }
infix operator >| : CompositionPrecedence

public func >| <A, B, C>(_ f: @escaping (A) -> B,
                  _ g: @escaping (B) -> C) -> (A) -> C {
    return { g(f($0)) }
}

//Image utils
//Convert a TensorPair to float tensor
public func toFloat<S1: TensorFlowScalar & Numeric, S2: TensorFlowScalar>(
    _ batch: TensorPair<S1,S2>
) -> TensorPair<Float,S2> {
    return TensorPair(input: Tensor<Float>(batch.input)/225.0, target: batch.target)
}

//Normalizes a TensorPair using mean and std
public struct Normalizer {
    public let mean: Tensor<Float>
    public let std: Tensor<Float>
    
    public init(mean: Tensor<Float>, std: Tensor<Float>) {
        (self.mean, self.std) = (mean, std)
    }
    
    public init(mean: Tensor<Float>, std: Tensor<Float>, expandingShapeAt: [Int], eps: Float = 1e-7) {
        self.mean = mean.expandingShape(at: expandingShapeAt)
        self.std  = std.expandingShape(at: expandingShapeAt) + eps
    }
    
    public func callAsFunction<S: TensorFlowScalar>(_ x: TensorPair<Float,S>) -> TensorPair<Float,S> {
        return TensorPair(input: (x.input - mean) / std, target: x.target)
    }
    
    public func denormalize<S: TensorFlowScalar>(_ x: TensorPair<Float,S>) -> TensorPair<Float,S> {
        return TensorPair(input: (x.input * std) + mean, target: x.target)
    }
}

//A normalizer using ImageNet statistics
public let imageNetNormalizer = Normalizer(
    mean: Tensor<Float>([0.485, 0.456, 0.406]),
    std:  Tensor<Float>([0.229, 0.224, 0.225]),
    expandingShapeAt: [0,1,2]
)

// Text utils
//Pad one text in x to length, using padIndex and padding at the beginning or the end
private func padOne(_ x: Tensor<Int32>, to length: Int, padIndex: Int = 1, padFirst: Bool = true) -> Tensor<Int32> {
    let pad: Tensor<Int32> = Tensor<Int32>(repeating: Int32(padIndex), shape: [length - x.shape[0]])
    return Tensor<Int32>(concatenating: padFirst ? [pad, x] : [x, pad])
}

//Returns a function to pad the inputs in TensorPair using padIndex and padFirst
public func padInputs<S: TensorFlowScalar>(padIndex: Int = 1, padFirst: Bool = true
) -> ([TensorPair<Int32,S>]) -> [TensorPair<Int32,S>] { 
    return { samples in
        let maxLength = samples.map { $0.input.shape[0] }.max()!
        return samples.map { TensorPair(input: padOne($0.input, to: maxLength), target: $0.target) }
    }
}
