
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class Data:
    def __init__(self, file):
        with open(file, 'r', encoding='utf-8') as json_file:
            self._data: dict = json.load(json_file)
            print(self._data)
        self._parsed_data, self._vectorizer = self.vectorize(self.extract(self._data))

    @property
    def parsed_data(self):
        return self._parsed_data

    def vectorizer(self):
        return self._vectorizer

    @staticmethod
    def extract(data_dict: dict) -> list:
        vid_list = []
        for item in data_dict:
            if item["details"]["name"] == "From Google Ads":
                vid_list.append("Ad")
            else:
                string = item["title"] + '' + item["subtitles"]["name"]
                vid_list.append(string)
        return vid_list

    @staticmethod
    def vectorize(vid_list:list):
        vectorizer = TfidfVectorizer(stop_words='english')
        transformed_data = vectorizer.fit_transform(vid_list)
        return transformed_data, vectorizer


class Cluster:

    def __init__(self, cluster_centers, num_clusters):
        self._cluster_centers = cluster_centers
        self.num_clusters = num_clusters

    @classmethod
    def k_means(cls, vector_data, num_clusters=4):
        model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500, random_state=42)
        model.fit(vector_data)
        cluster_centers = model.cluster_centers_
        return Cluster(cluster_centers, num_clusters)

    def print_top_terms_per_cluster(self, vectorizer, words=5):
        print("Top terms per cluster:")
        order_centroids = self._cluster_centers.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(self.num_clusters):
            print("Cluster %d:" % i, end=" ")
            for ind in order_centroids[i, :words]:
                print(' %s' % terms[ind], end=" ")
            print()
