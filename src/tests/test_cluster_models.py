"""Unit tests for functions in `src.code.cluster_models`.
"""

import unittest
from src.code.cluster_models import *


class TestClusterModels(unittest.TestCase):

    def setUp(self):
        self._data = [{"details": "ad"}, {"details": "ad", "subtitles":[{"name": "1"}], "title": "1"},
                {"subtitles":[{"name": "2"}], "title": "2"}, {"subtitles":[{"name": "3"}], "title": "3"},
                {"subtitles":[{"name": "2"}], "title": "2", "details": "ad"}]

    def test_extract(self):
        extract_list = ["Ad", "Ad", "2 2", "3 3", "Ad"]
        self.assertEqual(extract_list, Data.extract(self._data))


if __name__ == '__main__':
    unittest.main()
