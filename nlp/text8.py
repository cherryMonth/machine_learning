from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec

sentences = word2vec.Text8Corpus("text8")   # 加载语料
model = word2vec.Word2Vec(sentences, size=64, min_count=5)
