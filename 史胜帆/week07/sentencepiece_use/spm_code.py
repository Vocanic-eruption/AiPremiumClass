# 使用 sentencepiece 特殊语料分词工具  生成专有名词文档的分词模型 并 形成对应的词表
# 用之前的红楼梦.txt为例子
import sentencepiece as spm

#引入外部文本 形成分词模型
#pm.SentencePieceTrainer.Train(input = 'hlm_c.txt',model_prefix = 'hlm_mod',vocab_size = 10000)


 # 加载生成的分词模型 用其进行文本分词
 # 1 实例化分词模型
sp = spm.SentencePieceProcessor(model_file = 'hlm_mod.model')
# 2 测试分词
res = sp.EncodeAsPieces('列位看官：你道此书从何而来？说起根由，虽近荒唐，细按则深有趣味。')
print(res)