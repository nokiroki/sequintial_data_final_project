from src import TransactionsDatasetWithEmbed
from src import Conv1dEmbedAutoEncoder

import unittest
import torch
import pandas as pd


class TestDataSet(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.read_csv('data/data/exp3_train_small.csv', index_col=[0])
        self.datamodule = TransactionsDatasetWithEmbed('data/data/exp3_train_small.csv', 40, False, True)

    def test_length(self):
        self.assertEqual(len(self.datamodule) * 40, self.df.shape[0])

    def test_shapes(self):
        test_scaled, test_embed = self.datamodule[0]
        with self.subTest():
            self.assertEqual(test_scaled.shape, (3, 40))

        with self.subTest():
            self.assertEqual(test_embed.shape, (40,))


class TestConv1dEmbed(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Conv1dEmbedAutoEncoder(4, 8)
        self.x = torch.randn(32, 3, 40)
        self.c = torch.randint(204, (32, 40))

    def test_output_shape(self):
        with torch.no_grad():
            latent = self.model(self.x, self.c)

        self.assertEqual(latent.shape, (32, 8, 28))

    def test_all_good(self):
        with torch.no_grad():
            self.model.predict_step(self.x, self.c)


if __name__ == '__main__':
    calc_test_suite = unittest.TestSuite()
    calc_test_suite.addTest(unittest.makeSuite(TestDataSet))
    calc_test_suite.addTest(unittest.makeSuite(TestConv1dEmbed))
    print(f'Amount of tests - {calc_test_suite.countTestCases()}')

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(calc_test_suite)
