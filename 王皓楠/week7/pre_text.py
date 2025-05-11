import sentencepiece as spm
import jieba
import sys
import io
#引入外部文本资料训练分词模型
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
spm.SentencePieceTrainer.Train(input='xi_you.txt', 
                                model_prefix='xi_you_mod',
                               vocab_size=20000) # Train the model

# 加载模型进行分词
sp = spm.SentencePieceProcessor(model_file='xi_you_mod.model')
# 测试分词
print(sp.EncodeAsPieces('混沌未分天地乱，茫茫渺渺无人见。自从盘古破鸿蒙，开辟从兹清浊辨。 覆载群生仰至仁，发明万物皆成善。欲知造化会元功，须看西游释厄传')) # 分词
print(jieba.lcut('混沌未分天地乱，茫茫渺渺无人见。自从盘古破鸿蒙，开辟从兹清浊辨。 覆载群生仰至仁，发明万物皆成善。欲知造化会元功，须看西游释厄传'))