# -*- coding: utf-8 -*-
#! /usr/bin/env python

import os, codecs
import gensim
import numpy as np
from random import gammavariate
from random import random
from gensim.utils import tokenize
from gensim.corpora import Dictionary

def doc_processing(documents, stopwordsFilePath='', thres=10, doc=False, dicPath=False):

    # read the stopwords file
    if stopwordsFilePath != '':
        file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
        stopwords = [line.strip() for line in file]
        file.close()

    if dicPath:
        dictionary = Dictionary.load('../dictionary_2corpora.dic')

        # file = codecs.open(dicPath, 'r', 'utf-8')
        # dictionary = [line.strip() for line in file]
        # file.close()

    N = len(documents)
    # print('%d texts' % N)
    wordCounts = []
    word2id = {}
    id2word = {}
    currentId = 0
    my_punctuation = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'

    # generate the word2id and id2word maps and count the number of times of words showing up in documents
    # bigram = gensim.models.Phrases(documents)
    # documents = bigram[documents]

    documents_=[]
    for i, document in enumerate(documents):
        # if i%1000 == 0:
        #     print('Document #%d ...' % i)

        if doc == False:
            words_in_sent = tokenize(document, deacc=False)

            wordCount = {}
            for word in words_in_sent:
                if len(word) > 1 and word not in stopwords and word not in my_punctuation:
                    if word not in word2id.keys():
                        word2id[word] = currentId
                        id2word[currentId] = word
                        currentId += 1
                    if word in wordCount:
                        wordCount[word] += 1
                    else:
                        wordCount[word] = 1
            wordCounts.append(wordCount)
            i += 1

        else:
            doc = []
            words_in_sent = tokenize(document, deacc=False)
            if stopwordsFilePath != '':
                for word in words_in_sent:
                    if dicPath:
                        if word in dictionary:
                            doc.append(word)
                    elif len(word) > 1 and word not in stopwords and word not in my_punctuation:
                        doc.append(word)
            else:
                stopwords = ''
                for word in words_in_sent:
                    doc.append(word)
            documents_.append(doc)

    if doc == False:
        word2id_ = {}
        id2word_ = {}
        M = len(word2id)

        # generate the document-word matrix
        X = np.zeros([N, M], dtype=np.int8)
        for i in range(N):
            for word in wordCounts[i]:
                j = word2id[word]
                if wordCounts[i][word] < 0:  
                    wordCounts[i][word] = 0
                X[i, j] = wordCounts[i][word]

        # Remove outliers
        X2 = []
        for w in range(X.shape[1]):
            thres_up = X.shape[0]*10
            if thres <= np.sum(X[:, w]) < thres_up:
                X2.append(X[:, w])
                word = id2word[w]
                word2id_[word] = word2id[word]
                id2word_[len(X2)-1] = word

        X2 = np.array(X2)
        X2 = X2.T
        M = X2.shape[1]
        print('Dictionary size: %d' % M)

        return N, M, word2id_, id2word_, X2

    else:
        return documents_, stopwords


# def delete_verb(texts):
#     for sentence in texts:
#         tagged_sent = pos_tag(sentence.split())
#         verbs = [word for word, pos in tagged_sent if 'VB' in pos]


def merge_docs(*args):
    all_in_one = []
    for doc in args:
        infile = codecs.open(doc, 'r', 'utf-8')
        for line in infile:
            all_in_one.append(line)

    return np.array(all_in_one)


def open_texts(directory):
    todo = []
    lstDir = os.walk(directory)
    for root, dir, files in lstDir:
        for file in files:
            textPath = os.path.join(root, file)
            textFile = codecs.open(textPath, 'r', 'utf-8')
            noticias = list(textFile)
            todo.append(noticias)
            textFile.close()
    return todo
