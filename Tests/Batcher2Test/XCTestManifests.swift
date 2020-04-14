import XCTest

#if !os(macOS)
  public func allTests() -> [XCTestCaseEntry] {
    return [
      testCase(Batcher2Tests.allTests),
    ]
  }
#endif