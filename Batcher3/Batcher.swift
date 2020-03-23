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

/// Type that represents a template used to build a `Batcher`.
public protocol BatcherProtocol {
    associatedtype SourceDataSet: RandomAccessCollection where SourceDataSet.Element: Collatable
    
    /// Hook to customize the way indices are sampled at each iteration.
    func sampleIndices(on: SourceDataSet) -> [Int]
    
    /// Hook to customize how the shuffling is done.
    func shuffleIndices(on: inout SourceDataSet, indices: [Int]) -> [Int]
    
    /// Hook to add padding to the samples before they are collated.
    func padSamples(samples: [SourceDataSet.Element]) -> [SourceDataSet.Element]
}

/// Default implementations of the `BatcherTemplate` methods.
public extension BatcherProtocol {
    
    /// Returns the elements of `0..<dataset.count`.
    func sampleIndices(on dataset: SourceDataSet) -> [Int] {
        return Array(0..<dataset.count)
    }
    
    /// Returns `indices` shuffled.
    func shuffleIndices(on dataset: inout SourceDataSet, indices: [Int]) -> [Int] {
        return indices.shuffled()
    }
    
    /// Returns `samples`.
    func padSamples(samples: [SourceDataSet.Element]) -> [SourceDataSet.Element] {
        return samples
    }
}

/// A collection that splits a dataset in batches
public struct Batcher<SourceDataSet: RandomAccessCollection>: BatcherProtocol 
where SourceDataSet.Element: Collatable {
    public typealias Element = SourceDataSet.Element
    /// The template used to build the batches.
    public var dataset: SourceDataSet
    /// The size of each batch.
    public var batchSize: Int
    /// Optionally set a limit to the number of threads used.
    public var threadsLimit: Int? = nil
    /// If `true`, drop the last batch if it has less elements than `batchSize`.
    public var droppingLast: Bool = false
    
    var indices: [Int]
    
    /// Returns the number of batches contained in the `Batcher`.
    public var count: Int {
        let nSamples = dataset.count
        return nSamples / batchSize + (nSamples % batchSize == 0 || droppingLast ? 0 : 1)
    }
    
    public init(
        on dataset: SourceDataSet, 
        batchSize: Int, 
        threadsLimit: Int? = nil, 
        droppingLast: Bool = false
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.threadsLimit = threadsLimit
        self.droppingLast = droppingLast
        indices = []
        indices = sampleIndices(on: dataset)
    }
}

extension Batcher: Collection {
    public typealias Index = Int 
    public var startIndex: Int { return 0 }
    public var endIndex: Int { 
        let n = dataset.count
        return n / batchSize + (n % batchSize == 0 || droppingLast ? 0 : 1)
    }
    
    public func index(after i: Int) -> Int { i+1 }
    
    /// Access the i-th batch
    public subscript(i: Int) -> Element {
        get {
            let start = i * batchSize
            let end = Swift.min(start+batchSize, dataset.count)
            return withDevice(.cpu) { () -> Element in
                let n = threadsLimit == nil ? 1 : (end - start) / threadsLimit!
                let samples = Array(start..<end).concurrentMap(minBatchSize: n) {
                    dataset[dataset.index(dataset.startIndex, offsetBy: indices[$0])]
                }
                return Element(collating: padSamples(samples: samples))
            }
        }
    }
    
    /// Sample the internal `indices` used, optionally `shuffled`.
    ///
    /// - Note: This should be called before each new epoch. 
    public mutating func reorder(shuffled: Bool = false) { 
        indices = sampleIndices(on: dataset)
        if shuffled { indices = shuffleIndices(on: &dataset, indices: indices) }
    }
}