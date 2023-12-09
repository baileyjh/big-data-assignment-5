# Assignment 5: Information Retrieval from Real Data

**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Link to Presentation



## Author Information
* **Name**: Bailey Hall
* **Email**: bahall@westmont.edu

## License Information

MIT License

Copyright (c) 2023 baileyjh

## Instructions for Software Utilization

This software is meant to cluster YouTube watch history data in the form of json file using the K-means flat clustering
algorithm and tools from sklearn. By using this program, one can get a rough estimate on what the n different types
of YouTube videos are that one has watched.

To utilize this program, navigate to src.code.cluster_runner.py. There is only one function, main(), in this file. In 
the function, there are three variables that can be changed to adjust the output of this program to the needs of the user.

The variable file_path should be set to the path to the YouTube watch history json file to be analyzed. The variable 
cluster_number should be set to the desired number of clusters the data will be assigned to. The variable top_terms
should be set to the desired number of top terms displayed from each cluster. The terms come from a combination of the
terms in a video's title and the name of the channel the video is from.

After those three variables are set, all that needs to be done is to run cluster_runner.py and see the output displayed
in the terminal. The output should have the same number of rows as clusters specified, while each row contains the 
number of videos in the cluster and the top terms contained in the cluster (the number of terms is dependent on 
specifications as well).

If more tweaking to the program is desired, the core logic for computing the clusters and vectorizing the text representing
a video is contained in src.code.cluster_models.py.

## Citations

1) https://devqa.io/python-parse-json/
2) https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
3) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
4) https://www.geeksforgeeks.org/clustering-text-documents-using-k-means-in-scikit-learn/
5) https://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
6) https://chat.openai.com/

ChatGPT Prompt: "Given the print code from citation 5, is there a way for me to print also the number of documents 
categorized per cluster?"
