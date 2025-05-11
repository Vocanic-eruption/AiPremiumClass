def cal_jaccard_distance(string1, string2):
    words1 = set(string1)
    words2 = set(string2)
    distance = len(words1 & words2) / len(words1 | words2)
    return distance
