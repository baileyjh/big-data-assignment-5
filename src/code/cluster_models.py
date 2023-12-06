
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


class Data:
    def __init__(self, file):
        with open(file) as json_file:
            self._data: dict = json.load(json_file)
            print(self._data)
        self._parsed_data = self.vectorize(self, self.extract(self, self._data))

    @property
    def parsed_data(self):
        return self._parsed_data

    @staticmethod
    def extract(self, data_dict: dict) -> list:
        vid_list = []
        for item in data_dict:
            if item["details"]["name"] == "From Google Ads":
                vid_list.append("Ad")
            else:
                string = item["title"] + item["subtitles"]["name"]
                vid_list.append(string)
        return vid_list

    @staticmethod
    def vectorize(self, vid_list:list):
        vectorizer = TfidfVectorizer(stop_words='english')
        transformed_data = vectorizer.fit_transform(vid_list)
        return transformed_data


class Cluster:

    def __init__(self, data):
        self._data = data