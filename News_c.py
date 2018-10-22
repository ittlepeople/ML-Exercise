import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models, similarities
import pandas as pd
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


if __name__ == '__main__':
    news = pd.read_table('data/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
    news = news.dropna()
    print(news.shape)

    '''
    ==============分词：使用结吧分词器======================
    '''
    # 转成list
    contents = news.content.values.tolist()
    contents_s = []
    for line in contents:
        # 分词
        current_segment = jieba.lcut(line)
        if len(current_segment) > 1 and current_segment != '\r\n':  # 换行符
            contents_s.append(current_segment)
    df_contents_s = pd.DataFrame({'contents_s': contents_s})
    stopwords = pd.read_csv('data/stopwords.txt', index_col=False, sep='/t', quoting=3, names=['stopword'], encoding='utf-8')

    contents = df_contents_s.contents_s.values.tolist()
    stopwords = stopwords.stopword.values.tolist()
    contents_clean, all_words = drop_stopwords(contents, stopwords)

    df_contents_clean = pd.DataFrame({'contents_clean': contents_clean})
    df_all_words = pd.DataFrame({'all_words': all_words})

    # groupby 以什么为基本。agg() 可以加函数，字符串，字典或字符串/函数列表
    # 两种方法有区别，注意体会
    # words_count = df_all_words['all_words'].groupby(by=df_all_words['all_words']).agg({'count': np.size})
    words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({'count': np.size})

    # 重置索引,旧索引将作为列添加,可以使用drop参数来避免将旧索引添加为列
    words_count = words_count.reset_index().sort_values(by=['count'], ascending=False)

    '''
    ================制作词云======================================================================
    '''
    matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

    wordcloud = WordCloud(font_path="./data/simhei.ttf", background_color="white", max_font_size=80)
    word_frequence = {x[0]: x[1] for x in words_count.head(100).values}
    wordcloud = wordcloud.fit_words(word_frequence)
    plt.imshow(wordcloud)

    '''
    ================ TF-IDF ：提取关键词=======================================================
    '''
    # str.join(sequence)  sequence中用str连接
    index = 2400
    print(news['content'][index])
    contents_s_str = ''.join(contents_s[index])
    print('  '.join(jieba.analyse.extract_tags(contents_s_str, topK=5, withWeight=False)))

    '''
    ==================LDA ：主题模型============================================================
    '''
    # 做映射，相当于词袋  输入为ist of list形式
    # 单词及其整数id之间的映射。可以理解为python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
    dictionary = corpora.Dictionary(contents_clean)
    # 将文档转换为词袋（BoW）格式= （token_id，token_count）元组的列表。
    # doc2bow（document，allow_update = False，return_missing = False ） 输入为list of str
    corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)  # 类似Kmeans自己指定K值
    print(lda.print_topic(1, topn=5))
    for topic in lda.print_topics(num_topics=20, num_words=5):
        print(topic[1])

    df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': news['category']})
    label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
    df_train['label'] = df_train['label'].map(label_mapping)
    # test_size: float, int or None, optional(default=0.25)
    x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)

    words = []
    for line_index in range(len(x_train)):
        try:
            # x_train[line_index][word_index] = str(x_train[line_index][word_index])
            words.append(' '.join(x_train[line_index]))
        except:
            print('异常：', line_index)
    print('words的长度：', len(words))

    # texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
    # #  将文本文档集合转换为令牌计数矩阵
    # cv = CountVectorizer()
    # # 学习词汇词典并返回术语 - 文档矩阵。
    # cv_fit = cv.fit_transform(texts)
    # # 从要素整数索引到要素名称的数组映射
    # print((cv.get_feature_names()))
    # print(cv_fit.toarray())

    '''
    =============使用CountVectorizer=======================
    '''
    cev = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
    # 学习原始文档中所有标记的词汇表
    cev.fit(words)
    '''
    用于多项式模型的朴素贝叶斯分类器   
    '''
    classifier = MultinomialNB()
    '''
    transform（raw_documents ）
    将文档转换为文档术语矩阵,使用符合fit的词汇表或提供给构造函数的词汇表，从原始文本文档中提取令牌计数。
    返回为：文档术语矩阵
    '''
    classifier.fit(cev.transform(words), y_train)

    test_words = []
    for line_index in range(len(x_test)):
        try:
            test_words.append(' '.join(x_test[line_index]))
        except:
            print('异常：', line_index)
    # 返回给定测试数据和标签的平均准确度
    score = classifier.score(cev.transform(test_words), y_test)
    print('数量矢量器朴素贝叶斯：', score)

    '''
    ==============使用tdidf矢量器===============
    '''
    vec = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vec.fit(words)
    classifier2 = MultinomialNB()
    classifier2.fit(vec.transform(words), y_train)
    score2 = classifier2.score(vec.transform(test_words), y_test)
    print('tfidf矢量器贝叶斯得分：', score2)