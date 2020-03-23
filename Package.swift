// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "SwiftData",
    products: [
      .library( name: "Batcher", targets: ["Batcher"]),
      .library( name: "Batcher1", targets: ["Batcher1"]),
      .library( name: "Batcher2", targets: ["Batcher2"]),
      .library( name: "Batcher3", targets: ["Batcher3"])
    ],
    targets: [
      .target( name: "Batcher", path: "Batcher"),
      .target( name: "Batcher1", path: "Batcher1"),
      .target( name: "Batcher2", path: "Batcher2"),
      .target( name: "Batcher3", path: "Batcher3")
    ]
)
