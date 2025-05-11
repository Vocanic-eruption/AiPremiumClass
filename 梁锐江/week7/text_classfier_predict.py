import jieba
import sentencepiece as spm
import torch

from 梁锐江.week7.text_classfier import TextClassifier

if __name__ == '__main__':
    hidden_size = 128
    embedding_dim = 100
    nums_classes = 3
    epochs = 10

    # vocab = torch.load('./vocab_idx.pth')
    sp = spm.SentencePieceProcessor(model_file='comments_spm.model')
    vocab = {sp.IdToPiece(i): i for i in range(sp.vocab_size())}
    vocab['PAD'] = sp.vocab_size()

    # 测试模型
    comment1 = '整个系列，都不得不看'
    comment2 = '第一个镜头居然以为小男主是女孩，，，真够温柔清秀的。。。。'

    # comment1 = torch.tensor([vocab[word] for word in jieba.lcut(comment1)]).unsqueeze(0)
    comment1 = torch.tensor([vocab[word] for word in sp.encode_as_pieces(comment1)]).unsqueeze(0)
    # comment2 = torch.tensor([vocab[word] for word in jieba.lcut(comment2)]).unsqueeze(0)
    comment2 = torch.tensor([vocab[word] for word in sp.encode_as_pieces(comment2)]).unsqueeze(0)

    model = TextClassifier(len(vocab), hidden_size, embedding_dim, nums_classes)
    model.load_state_dict(torch.load('./text_classifier.pth'))

    model.eval()

    pre1 = model(comment1)
    pre2 = model(comment2)

    result = torch.argmax(pre1, dim=1).item()
    result2 = torch.argmax(pre2, dim=1).item()
    print(f'评论1预测结果{result}')
    print(f'评论2预测结果{result2}')
