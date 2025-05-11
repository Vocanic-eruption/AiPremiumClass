
def tokenize_pkuseg(text):
    import pkuseg
    seg = pkuseg.pkuseg()
    return seg.cut(text)

def tokenize_thulac(text):
    import thulac
    thu = thulac.thulac(seg_only=True)
    return thu.cut(text, text=True).split()

def evaluate_tokenizer(tokenizer_func):

    df['tokens'] = df['cleaned'].apply(tokenizer_func)
    X = [text_to_indices(tokens) for tokens in df['tokens']]
    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train)), batch_size=64)
    
    
    model = TextClassifier(len(vocab))
    optimizer = optim.Adam(model.parameters())
    for _ in range(5): 
        train()
    
  
    return evaluate(DataLoader(TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test)), batch_size=64))


print("\n不同分词工具准确率比较:")
print(f"Jieba: {test_acc:.4f}")  
print(f"PKUSeg: {evaluate_tokenizer(tokenize_pkuseg):.4f}")
print(f"THULAC: {evaluate_tokenizer(tokenize_thulac):.4f}")
