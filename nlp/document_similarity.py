from jieba import analyse
import numpy
import gensim


# 提取关键词
def keyword_extract(data):
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords


# 将文档的每句话进行关键词提取，并将结果保存在txt文件中
def getKeywords(docpath):
    keyword_string_list = list()
    with open(docpath, 'r', encoding='utf-8') as docf:
        for data in docf:  # 以换行划分段落，一个data是一段
            keywords_string = '/'.join(keyword_extract(data))
            keyword_string_list.append(keywords_string)
    return keyword_string_list


def word2vec(keywords_string_list, model):
    wordvec_size = 64
    word_vec_all = numpy.zeros(wordvec_size)
    keyword_num = 0
    for keywords_string in keywords_string_list:
        keywords_list = keywords_string.split("/")
        for word in keywords_list:
            if word in model:
                keyword_num += 1
                word_vec_all = word_vec_all + model[word]

    return word_vec_all / keyword_num


# 词向量相似度计算代码：余弦
def simlarityCalu(vector1, vector2):
    vector1Mod = numpy.sqrt(vector1.dot(vector1))
    vector2Mod = numpy.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


p1 = 'untitled.txt'
p2 = 'untitled1.txt'
# 获取关键词
p1_keywords = getKeywords(p1)
p2_keywords = getKeywords(p2)
print(p1_keywords)
print(p2_keywords)
model_file = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
p1_vec = word2vec(p1_keywords, model)
p2_vec = word2vec(p2_keywords, model)
# 计算相似度
print(simlarityCalu(p1_vec, p2_vec))
