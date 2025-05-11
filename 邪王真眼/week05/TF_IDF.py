import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

from utils.tfidf_calculator import TFIDFCalculator
from sklearn.metrics.pairwise import cosine_similarity


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)


def load_and_preprocess_comments(file_path):
    comments = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        prev_book_name = ''
        for i, line in enumerate(lines):
            if i == 0:
                continue

            terms = line.split("\t")
            if len(terms) >= 6:
                if terms[0] == '':
                    terms[0] = prev_book_name
                if terms[0] not in comments:
                    comments[terms[0]] = []
                
                comment_text = "\t".join(terms[5:])
                comments[terms[0]].append(comment_text)
                prev_book_name = terms[0]
            else:
                comments[prev_book_name][-1] += terms[0]
            
    return comments


def get_most_similar_books(tfidf_matrix, book_names, selected_book, top_n=5):
    selected_idx = book_names.index(selected_book)
    similarities = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()
    similarities[selected_idx] = -1
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(book_names[i], similarities[i]) for i in top_indices]


def main():
    dataset_path = './邪王真眼/datasets/douban_comments_top250/'
    comments_file = os.path.join(dataset_path, 'doubanbook_top250_comments.txt')
    stopwords_file = os.path.join(dataset_path, 'stopwords.txt')

    comments = load_and_preprocess_comments(comments_file)
    stopwords = load_stopwords(stopwords_file)
    
    selected_book = random.choice(list(comments.keys()))
    selected_book = '盗墓笔记'
    print(f"Selected Book: {selected_book}")

    manual = True

    if manual == True:
        tfidf, all_terms, _ = TFIDFCalculator.compute_tfidf(comments, stopwords)

        similar_books = get_most_similar_books(tfidf, all_terms, selected_book, top_n=5)

        print("Most similar books:")
        for book, similarity in similar_books:
            print(f"Book: {book}, Similarity: {similarity:.4f}")
    
    else:
        tfidf, all_terms, _ = TFIDFCalculator.compute_tfidf_by_sklearn(comments, stopwords)

        similar = cosine_similarity(tfidf)

        selected_book_index = list(comments.keys()).index(selected_book)
        similarity_scores = similar[selected_book_index]
        similarity_scores[selected_book_index] = -1
        most_similar_books_indices = similarity_scores.argsort()[-5:][::-1]
        
        print("Most similar books:")
        for index in most_similar_books_indices:
            book_name = list(comments.keys())[index]
            similarity = similarity_scores[index]
            print(f"Book: {book_name}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
