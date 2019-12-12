// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "Utils",
    products: [
      .library( name: "Utils", targets: ["Utils"]),
      .executable( name: "dataload", targets: ["dataload"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mxcl/Path.swift", from: "0.16.3"),
        .package(path: "../SwiftCV")
    ],
    targets: [
        .target( name: "Utils", dependencies: ["SwiftCV", "Path"]),
        .target( name: "dataload", dependencies: ["SwiftCV", "Path", "Utils"])
    ]
)
