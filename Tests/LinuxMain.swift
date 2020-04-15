import BatcherTest
import Batcher1Test
import Batcher2Test
import EpochsTest
import XCTest

var tests = [XCTestCaseEntry]()
//tests += BatcherTest.allTests()
//tests += Batcher1Test.allTests()
//tests += Batcher2Test.allTests()
tests += EpochsTest.allTests()
XCTMain(tests)