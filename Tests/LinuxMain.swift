import BatcherTest
import Batcher1Test
import XCTest

var tests = [XCTestCaseEntry]()
tests += BatcherTest.allTests()
tests += Batcher1Test.allTests()
XCTMain(tests)