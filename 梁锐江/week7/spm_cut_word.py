import sentencepiece as spm
import csv


def spm_before():
    with open('./comments.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        contents = []
        head_size = 0
        for i, line in enumerate(reader):
            content = line['content']
            if i == 0:
                head_size = len(line)
            if len(line) < head_size:
                continue
            else:
                contents.append(content)

    with open('./comments_cooking.txt', 'w', encoding='utf-8') as t:
        for line in contents:

            fixed_line = line.replace('\n', '')
            if len(fixed_line) < 5:
                continue
            t.write(fixed_line + '\n')


if __name__ == '__main__':
    # spm_before()
    spm.SentencePieceTrainer.Train(input='./comments_cooking.txt', model_prefix='comments_spm', vocab_size=7500)
    # sp = spm.SentencePieceProcessor(model_file='comments_spm.model')
    # print(sp.encode_as_pieces('看完以后，心变得很静'))
    # print(sp.IdToPiece(0))
