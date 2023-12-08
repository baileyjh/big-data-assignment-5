"""Runs the classifier on the selected corpus and displays the 30 "most informative features" of the document to
determine its class and the accuracy of the classifier.
"""

from src.code.cluster_models import Data, Cluster

__author__ = "Bailey Hall"
__copyright__ = "Copyright 2023, Westmont College, Bailey Hall"
__license__ = "MIT"
__email__ = "bahall@westmont.edu"


def main() -> None:
    file_path = '/Users/baileyhall/Desktop/BigData/assignment-5-IR/youtube-watch-history.json'
    data = Data(file_path)
    cluster = Cluster.k_means(data.vectorized_data, 10)
    cluster.print(data.vectorizer, 2)


main()
