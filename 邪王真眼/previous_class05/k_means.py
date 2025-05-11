import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math

from sklearn.cluster import KMeans
from collections import defaultdict

from utils.word2vec_vectorizer import GensimVectorizer


def main():
    model = GensimVectorizer(use_pretrained=True)
    
    with open("./邪王真眼/datasets/titles/titles.txt", encoding="utf8") as f:
        sentences = f.read().splitlines()
    
    vectors = model.batch_texts_to_vectors(sentences)

    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for label, sentences in sentence_label_dict.items():
        print(f"cluster {label} :")
        for i in range(min(5, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("-------------------------------------")


if __name__ == "__main__":
    main()
