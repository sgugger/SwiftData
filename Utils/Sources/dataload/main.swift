import Path
import TensorFlow
import SwiftCV
import Utils
import Foundation

public protocol BatcherConfig {
    associatedtype Item
    associatedtype Sample
    associatedtype RawBatch
    associatedtype Batch
    
    func processItem(_ item: Item) -> Sample
    func processSamples(_ samples: [Sample]) -> [Sample]
    func collate(_ samples: [Sample]) -> RawBatch 
    func processBatch(_ batch: RawBatch) -> Batch
}

public struct Batcher<C>: Sequence where C: BatcherConfig {
    public let config: C
    public let dataset: [C.Item]
    public var batchSize: Int
    public var numWorkers: Int = 1
    
    public init(_ config: C, dataset: [C.Item], batchSize: Int, numWorkers: Int = 1) {
        (self.config, self.dataset, self.batchSize, self.numWorkers) = (config, dataset, batchSize, numWorkers)
    }
    
    public func makeIterator() -> BatchIterator<C> {
        return BatchIterator(self)
    }
}

public struct BatchIterator<C>: IteratorProtocol where C: BatcherConfig {
    let config: C
    let dataset: [C.Item]
    let batchSize: Int
    let numWorkers: Int
    var idx: Int = 0
    
    init(_ b: Batcher<C>) {
        (self.config, self.dataset, self.batchSize, self.numWorkers) = (b.config, b.dataset, b.batchSize, b.numWorkers)
    }
    
    public mutating func next() -> C.Batch? {
        guard idx < dataset.count else { return nil }
        let end = min(idx + batchSize, dataset.count)
        let samples = Array(idx..<end).concurrentMap(nthreads: numWorkers) {
            config.processItem(dataset[$0])
        }
        idx = end
        let batch = config.collate(config.processSamples(samples))
        return config.processBatch(batch)
    }
}

let dataPath = Path.home/".fastai"/"data"

func openImage(_ fn: Path) -> Mat {
    return imdecode(try! Data(contentsOf: fn.url))
}

func BGRToRGB(_ img: Mat) -> Mat {
    return cvtColor(img, nil, ColorConversionCode.COLOR_BGR2RGB)
}

func resize(_ img: Mat, size: Int) -> Mat {
    return resize(img, nil, Size(size, size), 0, 0, InterpolationFlag.INTER_LINEAR)
}

func cvImgToTensor(_ img: Mat) -> Tensor<UInt8> {
    return Tensor<UInt8>(cvMat: img)!
}

let path = dataPath/"imagenette2"/"train"
let fnames = collectFiles(under: path, recurse: true, filtering: ["jpeg", "jpg"])

let processImage = openImage >| BGRToRGB >| { resize($0, size: 224) } >| cvImgToTensor

let allLabels = fnames.map { $0.parent.basename() }
let labels = Array(Set(allLabels)).sorted()
let labelToInt = Dictionary(uniqueKeysWithValues: labels.enumerated().map{ ($0.element, $0.offset) })

struct imageConfig: BatcherConfig {
    typealias Item = Path
    typealias Sample = (Tensor<UInt8>, Int)
    typealias RawBatch = (Tensor<UInt8>, Tensor<Int32>)
    typealias Batch = (Tensor<Float>, Tensor<Int32>)
    
    func processItem(_ item: Item) -> Sample {
        return (processImage(item), labelToInt[item.parent.basename()]!)
    }
    
    func processSamples(_ samples: [Sample]) -> [Sample] { return samples }
    
    func collate(_ samples: [Sample]) -> RawBatch {
        return (Tensor(concatenating: samples.map { $0.0.expandingShape(at: 0) }), Tensor(samples.map { Int32($0.1) }))
    }
    
    func processBatch(_ batch: RawBatch) -> Batch {
        return (Tensor<Float>(batch.0) / 255.0, batch.1)
    }
}

SetNumThreads(0)

let batcher = Batcher(imageConfig(), dataset: fnames, batchSize: 256, numWorkers: 4)
let t = Tensor<Float>(zeros: [1])

import Dispatch

let start = DispatchTime.now()
for b in batcher {
    var s = (b.0.shape, b.1.shape)
}
let end = DispatchTime.now()
let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
let seconds = nanoseconds / 1e9
print(seconds)
