import Foundation
import Path
import TensorFlow

// ThreadSafe and concurrentMap based on https://talk.objc.io/episodes/S01E90-concurrent-map
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

precedencegroup CompositionPrecedence { associativity: left }
infix operator >| : CompositionPrecedence

public func >| <A, B, C>(_ f: @escaping (A) -> B,
                  _ g: @escaping (B) -> C) -> (A) -> C {
    return { g(f($0)) }
}

//Image utils
public func toFloat<S1: TensorFlowScalar&Numeric, S2: TensorFlowScalar>(
    _ batch: (Tensor<S1>, Tensor<S2>)) -> (Tensor<Float>, Tensor<S2>) {
    return (Tensor<Float>(batch.0)/225.0, batch.1)
}

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
    
    public func callAsFunction<S: TensorFlowScalar>(_ x: (Tensor<Float>, Tensor<S>)) -> (Tensor<Float>, Tensor<S>) {
        return ((x.0 - mean) / std, x.1)
    }
    
    public func denormalize<S: TensorFlowScalar>(_ x: (Tensor<Float>, Tensor<S>)) -> (Tensor<Float>, Tensor<S>) {
        return ((x.0 * std) + mean, x.1)
    }
}

public let imageNetNormalizer = Normalizer(
    mean: Tensor<Float>([0.485, 0.456, 0.406]),
    std:  Tensor<Float>([0.229, 0.224, 0.225]),
    expandingShapeAt: [0,1,2]
)

// Text utils
private func padOne(_ x: Tensor<Int32>, to length: Int, padIndex: Int = 1, padFirst: Bool = true) -> Tensor<Int32> {
    let pad: Tensor<Int32> = Tensor<Int32>(repeating: Int32(padIndex), shape: [length - x.shape[0]])
    return Tensor<Int32>(concatenating: padFirst ? [pad, x] : [x, pad])
}

public func padInputs<S: TensorFlowScalar>(padIndex: Int = 1, padFirst: Bool = true
) -> ([(Tensor<Int32>, Tensor<S>)]) -> [(Tensor<Int32>, Tensor<S>)] { 
    return { samples in
        let maxLength = samples.map { $0.0.shape[0] }.max()!
        return samples.map { (padOne($0.0, to: maxLength), $0.1) }
    }
}
