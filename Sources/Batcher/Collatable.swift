import TensorFlow

//Private protocol used to derive conformance to Collatable using KeyPathIterable
public protocol _Collatable {
 static func _collateLeaf<Root>(
    _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root])
}

//Protocol to conform to to use the default collateSamples function
public protocol Collatable: _Collatable {
  init(collating: [Self])
}

extension Collatable {
  public static func _collateLeaf<Root>(
     _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root]) {
    guard let keyPath = rootKeyPath as? WritableKeyPath<Root, Self> else {
      fatalError(
        "Failed conversion from \(rootKeyPath) to 'WritableKeyPath<\(Root.self), \(Self.self)>'")
    }
    rootOut[keyPath: keyPath] = Self.init(collating: rootIn.map { $0[keyPath: keyPath] })
  }
}

extension _KeyPathIterableBase {
 public func _collateAll<Root>(
    _ rootOut: inout Root, _ rootKeyPath: PartialKeyPath<Root>, _ rootIn: [Root]) {
   for kp in _allKeyPathsTypeErased {
      let joinedKeyPath = rootKeyPath.appending(path: kp)!
      if let valueType = type(of: joinedKeyPath).valueType as? _Collatable.Type {
        valueType._collateLeaf(&rootOut, joinedKeyPath, rootIn)
      } else if let nested = self[keyPath: kp] as? _KeyPathIterableBase {
        nested._collateAll(&rootOut, joinedKeyPath, rootIn)
      } else {
        fatalError("Key path \(kp) is not Collatable")
      }
    }
 }
}

extension KeyPathIterable {
  public init(collating roots: [Self]) {
    self = roots[0]
    _collateAll(&self, \.self, roots)
  }
}

//Tensor are collated using stacking
extension Tensor: Collatable {
    public init(collating: [Self]) { self.init(stacking: collating) }
}

//Example: you can derive conformance to Collatable directly if a struct has only tensors
//struct Pair : Collatable, KeyPathIterable {
//  var first: Tensor
//  var second: Tensor
//  var third: Tensor = Tensor(5.0)
//}
