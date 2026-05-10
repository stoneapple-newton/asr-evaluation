import unittest

from codex_version.evaluators import aggregate_asr, evaluate_asr, normalize_text


class ASREvaluatorTests(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_text("Hello, WORLD!"), "hello world")
        evaluation = evaluate_asr("Hello, WORLD!", "hello world")
        self.assertEqual(evaluation.wer, 0.0)
        self.assertEqual(evaluation.cer, 0.0)
        self.assertEqual(evaluation.wip, 1.0)
        self.assertEqual(evaluation.wil, 0.0)

    def test_word_error_counts_and_rates(self) -> None:
        evaluation = evaluate_asr("the cat sat", "the bat sat down")
        self.assertEqual(evaluation.hits, 2)
        self.assertEqual(evaluation.substitutions, 1)
        self.assertEqual(evaluation.deletions, 0)
        self.assertEqual(evaluation.insertions, 1)
        self.assertAlmostEqual(evaluation.wer, 2 / 3)
        self.assertAlmostEqual(evaluation.mer, 2 / 4)

    def test_aggregate_asr_sums_segment_counts(self) -> None:
        first = evaluate_asr("hello world", "hello world")
        second = evaluate_asr("the cat sat", "the bat sat down")
        aggregate = aggregate_asr([first, second])
        self.assertEqual(aggregate.reference_words, 5)
        self.assertEqual(aggregate.hypothesis_words, 6)
        self.assertEqual(aggregate.substitutions, 1)
        self.assertEqual(aggregate.insertions, 1)
        self.assertAlmostEqual(aggregate.wer, 2 / 5)


if __name__ == "__main__":
    unittest.main()
