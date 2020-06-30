import unittest

from train import train_model


class TestTrain(unittest.TestCase):
    def test_sanity_check(self):
        max_avg_reward, benchmark_reward = train_model(
            num_iterations=3,
            batch_size=3,
            verbose='none',
            visualizer_type='none',
            persist_progress=False,
        )
        print('Final best reward achieved is {0} against benchmark reward {1}'.format(max_avg_reward, benchmark_reward))
        self.assertTrue(max_avg_reward > 0.0)
        self.assertTrue(benchmark_reward > 0.0)

    def test_training_effectiveness(self):
        max_avg_reward, benchmark_reward = train_model(
            num_iterations=400,
            batch_size=512,
            verbose='progress',
            visualizer_type='none',
            persist_progress=False,
        )
        print('Final best reward achieved is {0} against benchmark reward {1}'.format(max_avg_reward, benchmark_reward))
        self.assertTrue(max_avg_reward > benchmark_reward)


if __name__ == '__main__':
    unittest.main()
