"""Implementation of two classes to cluster YouTube watch history data. The class Data takes in a json file
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
    """Class that takes in a YouTube watch history json file and computes the TF-IDF vector for each video.

        The `__init__` constructor takes in a file path, opens the file as a dictionary, and then calls the extract()
                and vectorize() methods on the file.
        _vectorized_data: Property that is a getter for the vectorized data
        _vectorizer: Property that is a getter for the vectorizer itself
        _extract (list): Method that takes in the dictionary form of the data and returns a list with elements of the
                form "Ad" if a video is an ad, or a string concatenation of the video title and channel name.
        _vectorize: Method that takes in the extracted data list and utilizes TfidfVectorizer from
                sklearn.feature_extraction.text to vectorize the extracted data for each video and then returns both the
                vectorized data and the vectorizer itself.
        """
    def __init__(self, file):
        with open(file, 'r', encoding='utf-8') as json_file:
            self._data: dict = json.load(json_file)
        self._vectorized_data, self._vectorizer = self.vectorize(self.extract(self._data))

    @property
    def vectorized_data(self):
        return self._vectorized_data

    @property
    def vectorizer(self):
        return self._vectorizer

    @staticmethod
    def extract(data_dict) -> list:
        vid_list = []
        for item in data_dict:
            # Only Google Ads have a "details" key
            if "details" in item.keys():
                vid_list.append("Ad")
            else:
                if "subtitles" in item.keys():
                    string = item["title"] + ' ' + item["subtitles"][0]["name"]
                    vid_list.append(string)
        return vid_list

    @staticmethod
    def vectorize(vid_list:list):
        vectorizer = TfidfVectorizer(stop_words='english')
        transformed_data = vectorizer.fit_transform(vid_list)
        return transformed_data, vectorizer


class Cluster:
    """Class that creates clusters from vectorized data.

        The `__init__` constructor takes in the cluster centers, labels, and the number of clusters calculated by the
                class method, kmeans()
        _kmeans: Class method that takes in vectorized data and the desired number of clusters to be constructed. This
                method utilizes KMeans from sklearn.cluster to create a clustering model with set parameters. It then
                returns an instance of the Cluster class with the arguments: cluster centers, center labels, and the
                number of clusters.
        _print: Method that takes in a vectorizer and specified number of words and then prints the label of each
                cluster, the number of videos in each cluster, and the top n words in each cluster.
        """

    def __init__(self, cluster_centers, cluster_labels, num_clusters):
        self._cluster_centers = cluster_centers
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters

    @classmethod
    def k_means(cls, vector_data, num_clusters: int):
        model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500, random_state=42)
        model.fit(vector_data)
        cluster_centers = model.cluster_centers_
        cluster_labels = model.labels_
        return Cluster(cluster_centers, cluster_labels, num_clusters)

    def print(self, vectorizer, words: int):
        order_centroids = self._cluster_centers.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(self.num_clusters):
            # Counts the number of videos in a specific cluster
            cluster_size = sum(self.cluster_labels == i)
            print(f"Cluster {i} (Videos: {cluster_size}):", end=" ")
            # Gets the top n words in each cluster
            for ind in order_centroids[i, :words]:
                print(' %s' % terms[ind], end=" ")
            print('')
