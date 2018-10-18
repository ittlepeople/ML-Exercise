import re
import collections


# 把语料中的单词全部抽取出来, 转成小写, 并且去除单词中间的特殊符号
def words(text):
    return re.findall('[a-z]+', text.lower())  # 返回值是一个列表


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    n = len(word)
    # set 无序、不重复的集合
    return set([word[0: i] + word[i + 1:] for i in range(n)] +  # 删除
               [word[0: i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # 交换
               [word[0: i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # 替换
               [word[0: i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # 增加


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


# 输入为列表
def known(words):
    return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])


if __name__ == '__main__':
    NWORDS = train(words(open('data/big.txt').read()))
    # appl #appla #learw #tess #morw
    print(correct('knon'))
