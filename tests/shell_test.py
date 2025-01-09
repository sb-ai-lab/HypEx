import unittest

from hypex.utils.test_wrappers import BaseTest


class MyTestCase(BaseTest):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == "__main__":
    unittest.main()
