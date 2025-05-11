def all_cut(dict, sentence):
    results = []

    if not isinstance(dict, set):
        dict_set = set()
        for word in dict.keys():
            dict_set.add(word)
    else:
        dict_set = dict
    
    for i in range(len(sentence)):
        if sentence[0:i+1] in dict_set:
            if i + 1 == len(sentence):
                results.append([sentence[0:i+1]])
                return results
            sub_results = all_cut(dict_set, sentence[i+1:])
            for sub_result in sub_results:
                sub_result.insert(0, sentence[0:i+1])
            results.extend(sub_results)
    
    return results

def main():
    dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}
    
    sentence = "经常有意见分歧"

    target = [
        ['经常', '有意见', '分歧'],
        ['经常', '有意见', '分', '歧'],
        ['经常', '有', '意见', '分歧'],
        ['经常', '有', '意见', '分', '歧'],
        ['经常', '有', '意', '见分歧'],
        ['经常', '有', '意', '见', '分歧'],
        ['经常', '有', '意', '见', '分', '歧'],
        ['经', '常', '有意见', '分歧'],
        ['经', '常', '有意见', '分', '歧'],
        ['经', '常', '有', '意见', '分歧'],
        ['经', '常', '有', '意见', '分', '歧'],
        ['经', '常', '有', '意', '见分歧'],
        ['经', '常', '有', '意', '见', '分歧'],
        ['经', '常', '有', '意', '见', '分', '歧']
    ]
    
    output = all_cut(dict, sentence)
    
    output.reverse()
    for item in target:
        print(item)
    print("\r")
    for item in output:
        print(item)

if __name__ == "__main__":
    main()