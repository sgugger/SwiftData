// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

/// Returns `x`.
public func identity<T>(_ x: T) -> T { x }

/// Returns the elements of `0..<dataset.count`, in that order if `shuffled == false`,
/// and randomly shuffled otherwise.
/// 
/// - Note: This is default implementation of `sampleIndices` in `Batcher`, which
/// is why `dataset` is inout.
public func defaultSample<SourceDataSet: RandomAccessCollection>(
    on dataset: inout SourceDataSet, shuffled: Bool) -> [Int] {
    return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
}


/// A collection that splits a dataset in batches.
public struct Batcher<SourceDataSet: RandomAccessCollection> 
where SourceDataSet.Element: Collatable {
    public typealias Element = SourceDataSet.Element
    /// The dataset to get the batches from.
    public var dataset: SourceDataSet
    /// The size of each batch.
    public var batchSize: Int
    /// Optionally set a limit to the number of threads used.
    public var threadsLimit: Int? = nil
    /// If `true`, shuffle the dataset at each iteration.
    public var shuffle: Bool = false
    /// If `true`, drop the last batch if it has less elements than `batchSize`.
    public var droppingLast: Bool = false
    /// Hook to customize the way indices are sampled at each iteration.
    public let sampleIndices: (inout SourceDataSet, Bool) -> [Int]
    /// Hook to add padding to the samples before they are collated.
    public let padSamples: ([Element]) -> [Element]
    
    /// Returns the number of batches contained in the `Batcher`.
    public var count: Int {
        let nSamples = dataset.count
        return nSamples / batchSize + (nSamples % batchSize == 0 || droppingLast ? 0 : 1)
    }
    
    public init(
        on dataset: SourceDataSet, 
        batchSize: Int, 
        threadsLimit: Int? = nil, 
        shuffle: Bool = false, 
        droppingLast: Bool = false,
        sampleIndices: @escaping (inout SourceDataSet, Bool) -> [Int] = defaultSample,
        padSamples: @escaping ([Element]) -> [Element] = identity
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.threadsLimit = threadsLimit
        self.shuffle = shuffle
        self.droppingLast = droppingLast
        self.sampleIndices = sampleIndices
        self.padSamples = padSamples
    }
    
    // To iterate through the batches
    public func sequenced() -> _BatchIterator<SourceDataSet> {
        return _BatchIterator(self)
    }
}

// Iterator through a Batcher
public struct _BatchIterator<SourceDataSet: RandomAccessCollection>: IteratorProtocol, Sequence 
where SourceDataSet.Element: Collatable {
    public typealias Element = SourceDataSet.Element
    /// Batcher to iterate through.
    var b: Batcher<SourceDataSet>
    /// Indices that will be used to go through the dataset of `b`.
    let indices: [Int]
    /// The length of the underlying dataset.
    let samplesCount: Int
    /// Where we are at in the dataset.
    var pos: Int = 0
    
    init(_ b: Batcher<SourceDataSet>) { 
        self.b = b
        indices = b.sampleIndices(&self.b.dataset, b.shuffle)
        samplesCount = b.dataset.count
        pos = 0
    }
    
    /// Returns the next batch
    public mutating func next() -> Element? {
        guard pos < samplesCount else { return nil }
        let end = Swift.min(pos + b.batchSize, samplesCount)
        if (end - pos) < b.batchSize && b.droppingLast { return nil }
        // The idea is to have samples processed and collated on the CPU before moving to the host.
        // This part has not been optimized yet
        return withDevice(.cpu) { () -> Element in
            let n = b.threadsLimit == nil ? 1 : (end-pos) / b.threadsLimit!
            let samples = Array(pos..<end).concurrentMap(minBatchSize: n) {
                b.dataset[b.dataset.index(b.dataset.startIndex, offsetBy: indices[$0])]
            }
            pos = end
            return Element(collating: b.padSamples(samples))
        }
    }
}