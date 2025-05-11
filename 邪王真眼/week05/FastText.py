import random
import fasttext
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def train_fasttext_model(unsupervised_text_file_path, supervised_text_file_path):
    # model = fasttext.train_unsupervised(input=unsupervised_text_file_path, model='skipgram', dim=100)
    # model = fasttext.train_unsupervised(input=unsupervised_text_file_path, model='cbow', dim=100)
    model = fasttext.train_supervised(input=supervised_text_file_path, dim=100)
    return model


def save_embeddings_to_tensorboard(model, selected_words, log_dir):
    words = model.get_words()
    word_vectors = np.array([model.get_word_vector(word) for word in words])
    
    selected_vectors = []
    metadata = []
    for word in selected_words:
        if word in words:
            idx = words.index(word)
            selected_vectors.append(word_vectors[idx])
            metadata.append(word)
    
    selected_vectors = np.array(selected_vectors)
    
    writer = SummaryWriter(log_dir)
    
    writer.add_embedding(
        selected_vectors, metadata=metadata, tag="selected_words"
    )
    
    print(f"写入 {len(selected_vectors)} 个词向量到 TensorBoard")
    
    writer.close()


def get_random_similarities_and_plot(model, log_dir):
    vocab = model.get_words()
    
    random_word = random.choice(vocab)
    
    similar_words = model.get_nearest_neighbors(random_word, k=3)
    
    all_neighbors = model.get_nearest_neighbors(random_word, k=20)
    dissimilar_words = sorted(all_neighbors, key=lambda x: x[0])[:3]
    
    random_words = random.sample(vocab, 3)
    
    selected_words = [random_word]
    selected_words.extend([word for score, word in similar_words])
    selected_words.extend([word for score, word in dissimilar_words])
    selected_words.extend(random_words)
    
    print(f"随机选择的词: {random_word}")
    print("最相似的词:")
    for score, word in similar_words:
        print(f"    {word}: {score:.4f}")
    
    print("最不相似的词:")
    for score, word in dissimilar_words:
        print(f"    {word}: {score:.4f}")
    
    print("随机选择的词:")
    for word in random_words:
        print('    ' + word)
    
    save_embeddings_to_tensorboard(model, selected_words, log_dir)


def main():
    unsupervised_text_file_path = r"C:\Users\97647\Desktop\corpus.txt"
    supervised_text_file_path = r"C:\Users\97647\Desktop\cooking.stackexchange.txt" # 移动到英文路径

    model = train_fasttext_model(unsupervised_text_file_path, supervised_text_file_path)

    get_random_similarities_and_plot(model, log_dir='./邪王真眼/week05/run')


if __name__ == "__main__":
    main()
