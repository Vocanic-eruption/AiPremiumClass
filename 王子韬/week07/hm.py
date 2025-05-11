import pandas as pd
import numpy as np
import re
import jieba
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import tempfile

# 1. 加载数据
def load_data(file_path='DMSC.csv'):
    """
    加载豆瓣电影短评数据
    评论得分1～2表示negative取值：0，评论得分4～5代表positive取值：1
    """
    print("加载数据...")
    df = pd.read_csv(file_path)
    
    # 打印列名，用于调试
    print("CSV文件的列名:", df.columns.tolist())
    print("数据前5行:")
    print(df.head())
    
    # 数据清洗和标签转换
    df = df[df['Comment'].notna()]  # 删除评论为空的行，使用'Comment'列而非'comment'
    
    # 将评分转换为二分类标签: 1-2分为0(负面)，4-5分为1(正面)，使用'Star'列而非'rating'
    df = df[(df['Star'] <= 2) | (df['Star'] >= 4)]
    df['label'] = df['Star'].apply(lambda x: 1 if x >= 4 else 0)
    
    # 为了演示，我们只使用部分数据
    neg_samples = df[df['label'] == 0].sample(min(1000, sum(df['label'] == 0)), random_state=42)
    pos_samples = df[df['label'] == 1].sample(min(1000, sum(df['label'] == 1)), random_state=42)
    
    # 合并样本
    balanced_df = pd.concat([neg_samples, pos_samples])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"处理后数据集大小: {balanced_df.shape}")
    print(f"正面评论数量: {sum(balanced_df['label'] == 1)}")
    print(f"负面评论数量: {sum(balanced_df['label'] == 0)}")
    
    return balanced_df

# 2. 文本预处理
def preprocess_text(text):
    """文本预处理：去除特殊字符、数字等"""
    if isinstance(text, str):
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除URL
        text = re.sub(r'http\S+', '', text)
        # 去除数字
        text = re.sub(r'\d+', '', text)
        # 去除特殊字符和标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# 3. 分词方法
def tokenize_jieba(text):
    """使用jieba进行中文分词"""
    return ' '.join(jieba.cut(text))

# 4. SentencePiece模型训练和分词
def train_sentencepiece(texts, vocab_size=8000, model_type='unigram', model_prefix='spm_model'):
    """
    训练SentencePiece模型
    
    参数:
    - texts: 文本列表
    - vocab_size: 词表大小
    - model_type: 模型类型 (unigram, bpe)
    - model_prefix: 模型文件前缀
    
    返回:
    - spm_processor: 训练好的SentencePiece处理器
    """
    print(f"训练SentencePiece模型 (vocab_size={vocab_size}, model_type={model_type})...")
    
    # 将文本保存到临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for text in texts:
            f.write(text + '\n')
        corpus_file = f.name
    
    # 训练SentencePiece模型
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # 为中文设置高字符覆盖率
        normalization_rule_name='nmt_nfkc_cf',  # 标准化规则
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    
    # 加载训练好的模型
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # 删除临时文件
    os.remove(corpus_file)
    
    print(f"SentencePiece模型训练完成，词表大小: {sp.get_piece_size()}")
    return sp

def tokenize_sentencepiece(text, sp_processor):
    """使用SentencePiece进行分词"""
    if isinstance(text, str) and len(text.strip()) > 0:
        tokens = sp_processor.encode_as_pieces(text)
        return ' '.join(tokens)
    return ""

# 5. 模型训练和评估
def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='nb'):
    """训练模型并评估"""
    if model_type == 'nb':
        model = MultinomialNB()
        model_name = "朴素贝叶斯"
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, C=1.0)
        model_name = "逻辑回归"
    else:
        raise ValueError("不支持的模型类型")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['负面', '正面'])
    
    print(f"模型: {model_name}")
    print(f"准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model, accuracy

# 6. 比较不同分词方法
def compare_tokenizers(df):
    """比较不同分词方法的效果"""
    # 预处理文本
    print("预处理文本...")
    # 使用'Comment'列而非'comment'或'cleaned_text'
    df['cleaned_text'] = df['Comment'].apply(preprocess_text)
    
    # 划分训练集和测试集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # 训练SentencePiece模型 (不同配置)
    spm_unigram = train_sentencepiece(X_train_raw, vocab_size=8000, model_type='unigram', model_prefix='spm_unigram')
    spm_bpe = train_sentencepiece(X_train_raw, vocab_size=8000, model_type='bpe', model_prefix='spm_bpe')
    
    # 定义要比较的分词器
    tokenizers = {
        'jieba': tokenize_jieba,
        'sentencepiece_unigram': lambda x: tokenize_sentencepiece(x, spm_unigram),
        'sentencepiece_bpe': lambda x: tokenize_sentencepiece(x, spm_bpe)
    }
    
    # 定义要比较的模型
    models = ['nb', 'lr']
    
    results = {}
    
    for tokenizer_name, tokenizer_func in tokenizers.items():
        print(f"\n使用 {tokenizer_name} 分词...")
        
        # 对训练集和测试集应用分词
        try:
            X_train_tokenized = X_train_raw.apply(tokenizer_func)
            X_test_tokenized = X_test_raw.apply(tokenizer_func)
            
            # 构建TF-IDF特征
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(X_train_tokenized)
            X_test = vectorizer.transform(X_test_tokenized)
            
            print(f"词典大小: {len(vectorizer.get_feature_names_out())}")
            
            # 对每种模型进行训练和评估
            for model_type in models:
                print(f"\n{tokenizer_name} + {model_type}:")
                model, accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, model_type)
                
                # 存储结果
                if tokenizer_name not in results:
                    results[tokenizer_name] = {}
                results[tokenizer_name][model_type] = accuracy
                
        except Exception as e:
            print(f"使用 {tokenizer_name} 时出错: {str(e)}")
    
    # 打印比较结果
    print("\n不同分词工具和模型的准确率比较:")
    print("-" * 60)
    print(f"{'分词工具':<20} | {'朴素贝叶斯':<15} | {'逻辑回归':<15}")
    print("-" * 60)
    
    for tokenizer, scores in results.items():
        nb_acc = scores.get('nb', 0)
        lr_acc = scores.get('lr', 0)
        print(f"{tokenizer:<20} | {nb_acc:.4f} | {lr_acc:.4f}")
    
    # 找出最佳配置
    best_accuracy = 0
    best_config = None
    
    for tokenizer, scores in results.items():
        for model, accuracy in scores.items():
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (tokenizer, model)
    
    print("-" * 60)
    print(f"最佳配置: {best_config[0]} + {'朴素贝叶斯' if best_config[1] == 'nb' else '逻辑回归'}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    return results

# 7. 测试几个示例评论
def test_examples(model, vectorizer, tokenizer_func):
    """测试几个示例评论"""
    print("\n测试示例评论:")
    examples = [
        "这部电影太棒了，演员的表演非常精彩，剧情扣人心弦",
        "画面不错，但是剧情太差了，浪费时间",
        "特效一般，演员演技尚可，整体中规中矩",
        "剧情老套，毫无新意，浪费票钱",
        "配乐非常出色，节奏把控得当，是今年最好看的电影之一"
    ]
    
    for example in examples:
        # 预处理
        cleaned = preprocess_text(example)
        # 分词
        tokenized = tokenizer_func(cleaned)
        # 向量化
        features = vectorizer.transform([tokenized])
        # 预测
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        print(f"评论: {example}")
        print(f"预测: {'正面' if prediction == 1 else '负面'}")
        print(f"概率: 负面 {probability[0]:.4f}, 正面 {probability[1]:.4f}")
        print("-" * 50)

# 8. 主函数
def main(file_path='DMSC.csv'):
    # 加载数据
    df = load_data(file_path)
    
    # 比较不同分词方法
    results = compare_tokenizers(df)
    
    # 找出最佳配置
    best_accuracy = 0
    best_config = None
    
    for tokenizer, scores in results.items():
        for model, accuracy in scores.items():
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (tokenizer, model)
    
    print(f"\n最佳配置: {best_config[0]} + {'朴素贝叶斯' if best_config[1] == 'nb' else '逻辑回归'}")
    
    # 使用最佳配置在完整数据集上重新训练
    print("\n使用最佳配置进行最终模型训练...")
    
    # 预处理文本
    df['cleaned_text'] = df['Comment'].apply(preprocess_text)  # 使用'Comment'列
    
    # 划分训练集和测试集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # 根据最佳配置选择分词器
    tokenizer_name = best_config[0]
    model_type = best_config[1]
    
    if tokenizer_name == 'jieba':
        tokenizer_func = tokenize_jieba
    elif tokenizer_name == 'sentencepiece_unigram':
        spm_model = train_sentencepiece(X_train_raw, vocab_size=8000, model_type='unigram', model_prefix='spm_final_unigram')
        tokenizer_func = lambda x: tokenize_sentencepiece(x, spm_model)
    elif tokenizer_name == 'sentencepiece_bpe':
        spm_model = train_sentencepiece(X_train_raw, vocab_size=8000, model_type='bpe', model_prefix='spm_final_bpe')
        tokenizer_func = lambda x: tokenize_sentencepiece(x, spm_model)
    
    # 分词
    X_train_tokenized = X_train_raw.apply(tokenizer_func)
    X_test_tokenized = X_test_raw.apply(tokenizer_func)
    
    # 构建TF-IDF特征
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_tokenized)
    X_test = vectorizer.transform(X_test_tokenized)
    
    # 训练最终模型
    final_model, _ = train_and_evaluate(X_train, y_train, X_test, y_test, model_type)
    
    # 测试几个示例评论
    test_examples(final_model, vectorizer, tokenizer_func)

if __name__ == "__main__":
    # 提供文件的绝对路径或相对路径
    main('DMSC.csv')  # 替换为你的实际文件路径