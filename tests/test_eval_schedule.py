import unittest

from utils.eval_schedule import should_evaluate_epoch


class EvalScheduleTest(unittest.TestCase):
    def test_default_start_keeps_existing_periodic_evaluation(self):
        self.assertTrue(should_evaluate_epoch(1, 1, 0))
        self.assertTrue(should_evaluate_epoch(5, 5, 0))
        self.assertFalse(should_evaluate_epoch(4, 5, 0))

    def test_positive_start_skips_epochs_before_threshold(self):
        self.assertFalse(should_evaluate_epoch(1, 1, 5))
        self.assertFalse(should_evaluate_epoch(4, 1, 5))
        self.assertTrue(should_evaluate_epoch(5, 1, 5))
        self.assertTrue(should_evaluate_epoch(6, 1, 5))

    def test_positive_start_preserves_eval_period(self):
        self.assertFalse(should_evaluate_epoch(5, 2, 5))
        self.assertTrue(should_evaluate_epoch(6, 2, 5))

    def test_negative_start_raises(self):
        with self.assertRaises(ValueError):
            should_evaluate_epoch(1, 1, -1)


if __name__ == "__main__":
    unittest.main()
