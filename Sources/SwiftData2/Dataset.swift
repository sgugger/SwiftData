import TensorFlow

public func basicDataset<Item, S1: TensorFlowScalar, S2: TensorFlowScalar> (
    from items:[Item], 
    toInput: @escaping (Item) -> Tensor<S1>,
    toTarget: @escaping (Item) -> Tensor<S2>) -> LazyMapSequence<[Item], (Tensor<S1>, Tensor<S2>)> {
    return items.lazy.map { (toInput($0), toTarget($0)) }
}

public struct LanguageModelDataset<Item>: Collection {
    public typealias Index = Int
    public typealias Element = (Tensor<Int32>, Tensor<Int32>)
    
    public let openItem: (Item) -> [Int]
    public var batchSize: Int
    public var sequenceLength: Int
    public var items: [Item]
    public var lengths: [Int]
    private var batchLength: Int
    private var batchCount: Int
    private var lastLength: Int
    private var cumLengths: [Int]
    //To conform to Collection
    public var startIndex: Int { return 0 }
    public var endIndex:   Int { return batchCount * batchSize }
    
    public init(openItem: @escaping (Item) -> [Int],
                batchSize: Int,
                sequenceLength: Int,
                items: [Item],
                lengths: [Int]) {
        (self.openItem, self.batchSize, self.sequenceLength) = (openItem, batchSize, sequenceLength)
        (self.items, self.lengths) = (items, lengths)
        batchLength = (lengths.reduce(0, +) - 1) / batchSize
        batchCount = batchLength / sequenceLength + (batchLength % sequenceLength == 0 ? 0 : 1)
        lastLength = batchLength - (batchCount - 1) * sequenceLength
        cumLengths = lengths.reduce(into: []) { $0.append(($0.last ?? 0) + $1) }
    }
    
    public init(openItem: @escaping (Item) -> [Int],
                batchSize: Int,
                sequenceLength: Int,
                items: [Item]) {
        self.init(openItem: openItem, batchSize: batchSize, sequenceLength: sequenceLength, items: items, 
             lengths: items.map { openItem($0).count })
    }
    
    // Method that returns the next index when iterating
    public func index(after i: Int) -> Int { return i+1 }

    // Required subscript for Collection
    public subscript(index: Int) -> Iterator.Element { get {
        let sampleLength = index / batchSize == batchCount - 1 ? lastLength : sequenceLength
        let startIndex = (index % batchSize) * batchLength + (index / batchSize) * sequenceLength
        let sample = readItems(from: startIndex, to: startIndex + sampleLength + 1)
        let sample32 = sample.map { Int32($0) }
        return (Tensor<Int32>(sample32.prefix(upTo: sampleLength)), 
                Tensor<Int32>(sample32.suffix(from: 1))) }
    }
    
    public func readItems(from start: Int, to end: Int) -> [Int] {
        var res: [Int] = []
        var index = cumLengths.firstIndex { $0 >= start }!
        var pos = start
        while pos < end {
            let x = openItem(items[index])
            let cumLen = ([0] + cumLengths)[index]
            let readFrom = pos - cumLen
            let readUntil = Swift.min(end - cumLen, x.count)
            res = res + Array(x[readFrom..<readUntil])
            pos = readUntil + cumLen
            index += 1
        }
        return res
    }
}