/// A simple pseudo-random number generator.
struct LinearCongruential: RandomNumberGenerator {
  /// The last value returned by `self.next`, or what `self` was seeded with.
  private var lastValue: UInt64

  /// Creates an instance with the given `seed`.
  ///
  /// Instances created with the same `seed` produce the same sequence of
  /// pseuedo-random results from `next()`.
  init(seed: UInt64 = 0) {
    lastValue = seed
  }

  /// Returns a value from a uniform, independent distribution of binary data.
  mutating func next() -> UInt64 {
    // A "good value" chosen from https://arxiv.org/pdf/2001.05304.pdf (Steele,
    // Guy; Vigna, Sebastiano (15 January 2020). "Computationally easy,
    // spectrally good multipliers for congruential pseudorandom number
    // generators". arXiv:2001.05304 [cs.DS].)
    let a: UInt64 = 0xaf251af3b0f025b5
    let c: UInt64 = 1
    lastValue = lastValue &* a &+ c
    return lastValue
  }
}

/// A random number generator that wraps and forwards its operations to an
/// instance of any type conforming to `RandomNumberGenerator`.
struct Randomness: RandomNumberGenerator {
  /// The wrapped generator.
  private var base: RandomNumberGenerator

  /// Creates an instance wrapping and forwarding operations to a copy of
  /// `base`.
  init(_ base: RandomNumberGenerator) {
    self.base = base
  }
  
  /// Creates an instance forwarding operations to the system random number
  /// generator.
  init() {
    base = SystemRandomNumberGenerator()
  }
  
  /// Returns a value from a uniform, independent distribution of binary data.
  mutating func next() -> UInt64 { base.next() }
}
