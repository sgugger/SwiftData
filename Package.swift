// swift-tools-version:5.2

import PackageDescription

let package = Package(
    name: "SwiftData",
    platforms: [ .macOS(.v10_13) ],
    products: [
        .library(name: "Batcher", targets: ["Batcher"]),
        .library(name: "Batcher1", targets: ["Batcher1"]),
        .library(name: "Batcher2", targets: ["Batcher2"]),
        .library(name: "Epochs", targets: ["Epochs"])
    ],
    targets: [
        .target(name: "Batcher", path: "Batcher"),
        .testTarget(
            name: "exampleBatcher",
            dependencies: ["Batcher"],
            path: "examples", sources: ["exampleBatcher.swift"]),
        .target(name: "Batcher1", path: "Batcher1"),
        .target(name: "Batcher2", path: "Batcher2"),
        .target(name: "Epochs", path: "Epochs")
    ])
