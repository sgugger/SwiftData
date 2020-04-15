let t0 = EpochsTests()
for (_, t) in EpochsTests.allTests {
  t(t0)()
}
