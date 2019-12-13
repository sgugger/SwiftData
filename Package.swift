// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "SwiftData",
    products: [
      .library( name: "SwiftData", targets: ["SwiftData"])
    ],
    dependencies: [
        .package(url: "https://github.com/mxcl/Path.swift", from: "0.16.3")
    ],
    targets: [
        .target( name: "SwiftData", dependencies: ["Path"])
    ]
)
