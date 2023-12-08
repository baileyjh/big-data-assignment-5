"""Implementation of two classes to cluster YouTube watch history data. The class Data takes in a YouTube json file
and vectorizes the data, while the class Cluster uses scikit-learn to cluster the vectors.
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

__author__ = "Bailey Hall"
__copyright__ = "Copyright 2023, Westmont College, Bailey Hall"
__license__ = "MIT"
__email__ = "bahall@westmont.edu"


class Data:
    """Class that

        The `__init__` constructor takes in
        _vectorized_data: Property that returns the
        _vectorizer: Property that returns the
        _extract (list): Method that
        _vectorize: Method that
        """
    def __init__(self, file):
        """Constructor

            :param file:
            """
        with open(file, 'r', encoding='utf-8') as json_file:
            self._data: dict = json.load(json_file)
        self._vectorized_data, self._vectorizer = self.vectorize(self.extract(self._data))

    @property
    def vectorized_data(self):
        """Getter for vectorized data"""
        return self._vectorized_data

    @property
    def vectorizer(self):
        """Getter for vectorizer"""
        return self._vectorizer

    @staticmethod
    def extract(data_dict: dict) -> list:
        """Method that

            :param data_dict:
            """
        vid_list = []
        for item in data_dict:
            if "details" in item.keys():
                vid_list.append("Ad")
            else:
                if "subtitles" in item.keys():
                    string = item["title"] + '' + item["subtitles"][0]["name"]
                    vid_list.append(string)
        return vid_list

    @staticmethod
    def vectorize(vid_list:list):
        """Method that

            :param vid_list:
            """
        vectorizer = TfidfVectorizer(stop_words='english')
        transformed_data = vectorizer.fit_transform(vid_list)
        return transformed_data, vectorizer


class Cluster:
    """Class that

        The `__init__` constructor takes in
        _kmeans: Class method that
        _print: Method that prints
        """

    def __init__(self, cluster_centers, cluster_labels, num_clusters):
        """Constructor

            :param cluster_centers:
            :param cluster_labels:
            :param num_clusters:
            """
        self._cluster_centers = cluster_centers
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters

    @classmethod
    def k_means(cls, vector_data, num_clusters=4):
        """Class method

            :param vector_data:
            :param num_clusters:
            """
        model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500, random_state=42)
        model.fit(vector_data)
        cluster_centers = model.cluster_centers_
        cluster_labels = model.labels_
        return Cluster(cluster_centers, cluster_labels, num_clusters)

    def print(self, vectorizer, words=3):
        """Method

            :param vectorizer:
            :param words:
            """
        order_centroids = self._cluster_centers.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(self.num_clusters):
            cluster_size = sum(self.cluster_labels == i)
            print(f"Cluster {i} (Videos: {cluster_size}):", end=" ")
            for ind in order_centroids[i, :words]:
                print(' %s' % terms[ind], end=" ")
            print('')
