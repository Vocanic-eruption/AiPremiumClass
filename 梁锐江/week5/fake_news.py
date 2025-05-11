import csv
import fasttext


def temp():
    text = row['text']
    label = row['label']
    train_data_text = []
    train_data_label = []
    test_data_text = []
    test_data_label = []
    # 测试数据
    test_data_text.append(text)
    test_data_label.append(label)
    # 训练数据
    train_data_text.append(text)
    train_data_label.append(label)


if __name__ == '__main__':
    with open('/kaggle/input/fake-news-classification/WELFake_Dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv.field_size_limit(1000000)
        count = 0
        train_lines = []
        test_lines = []
        for row in reader:
            text = row['text']
            label = row['label']
            line = '__label__' + label + ' ' + text
            if count % 5 == 0:
                test_lines.append({"text": text, "label": label})
            else:
                train_lines.append(line)
            count += 1

    with open('/kaggle/working/fake_news_convert.txt', 'w', encoding='utf-8') as convert:
        for c_text in train_lines:
            convert.write(c_text)

    with open('/kaggle/working/fake_news_convert_test.txt', 'w', encoding='utf-8') as convert_test:
        for c_test_text in test_lines:
            convert_test.write(c_text)

    model = fasttext.train_supervised('/kaggle/working/fake_news_convert.txt', epoch=10)
    print(model.get_word_vector('with'))

    act = 0
    for test_linie in test_lines:
        text = test_linie['text'].replace("\n", " ")
        label = test_linie['label']
        predict = model.predict(text)
        if predict[0][0][-1] == label:
            act += 1
        # print(f'预测值:{predict[0][0][-1]}, 实际值:{label}')
        # break
    print(f'准确率:{act / len(test_lines)}')
