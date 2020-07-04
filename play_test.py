"""
Tests for model testing
"""

import unittest

from play import test_model


class TestPlay(unittest.TestCase):
    """
    Test suite for model testing
    """

    def test_sanity_check(self):
        """
        Tests if the model testing finishes without error

        :return: None
        """
        avg_reward = test_model(persist_progress_option='none', verbose='none')
        self.assertTrue(avg_reward > 0.0)
