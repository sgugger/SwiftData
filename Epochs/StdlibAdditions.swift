extension Collection {
  /// Returns the `n`th position in `self`.
  func index(atOffset n: Int) -> Index { index(startIndex, offsetBy: n) }
}
