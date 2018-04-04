#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs, os, re
import warnings
import numpy as np

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from time import time

warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

########################################################################################
def build_texts(fname):
    with open(fname) as f:
        for i,line in enumerate(f):
            yield line.decode('utf-8')
    f.close()


def get_keywords(N, weights, words_in_file):
    idx2word = dict((idx,word) for word,idx in words_in_file.iteritems())
    # keywords = np.empty((N,1), dtype=basestring)
    keywords = ''
    weight_keywords = np.zeros((N,1), dtype='float32')
    for k in xrange(N):
        if k < (N-1):
            keywords += idx2word[np.argmax(weights)] + ', '
        else:
            keywords += idx2word[np.argmax(weights)]
        weight_keywords[k] = weights[np.argmax(weights)]
        weights[np.argmax(weights)] = -1.0
    return keywords, weight_keywords


def doc_processing(documents, stopwordsFilePath='', thres=10, doc=False, dicPath=False):

    # read the stopwords file
    if stopwordsFilePath != '':
        file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
        stopwords = [line.strip() for line in file]
        file.close()
    else:
        stopwords = ''

    if dicPath:
        dictionary = Dictionary.load('../dictionary_2corpora.dic')

        # file = codecs.open(dicPath, 'r', 'utf-8')
        # dictionary = [line.strip() for line in file]
        # file.close()

    N = len(documents)
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
            for word in words_in_sent:
                if dicPath:
                    if word in dictionary:
                        doc.append(word)
                elif len(word) > 1 and word not in stopwords and word not in my_punctuation:
                    doc.append(word)


            # if stopwordsFilePath != '':
            #     for word in words_in_sent:
            #         if dicPath:
            #             if word in dictionary:
            #                 doc.append(word)
            #         elif len(word) > 1 and word not in stopwords and word not in my_punctuation:
            #             doc.append(word)
            # else:
            #     stopwords = ''
            #     for word in words_in_sent:
            #         doc.append(word)
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
                if wordCounts[i][word] < 0:   # Esto no se porque pasa pero alguna vez suelta pone un numero negativo (random) si la palabra no aparece
                    wordCounts[i][word] = 0
                X[i, j] = wordCounts[i][word]

        # Elimino palabras en los extremos
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

def save_obj(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=2)

########################################################################################


def compute_keywords(X_train_raw, n_keywords=20, k_topics=50, alpha=0.3, eta=1, niter=200, ismain=False, news_index=[], news_name=[]):
    print('compute_keywords\n')
    stopwordsFilePath = '../stopwords_es_ES_enh.txt'
    my_punctuation = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'

    # X_train_raw = [article.content for article in newspaper]
    processed_docs, stopwords = doc_processing(X_train_raw, stopwordsFilePath, doc=True)
    id2word = Dictionary(processed_docs)
    # dictionary.filter_extremes(no_below=1, no_above=0.6, keep_n=None)
    corpus = [id2word.doc2bow(text) for text in processed_docs]

    # Topic modeling using LDA
    ldamodel = LdaModel(corpus=corpus, num_topics=k_topics, alpha=alpha, eta=eta, id2word=id2word, iterations=niter)
    num_words = 50
    ldatopics = ldamodel.top_topics(corpus, num_words=num_words)

    topic_word_matrix = ldamodel.expElogbeta
    word2id = id2word.token2id

    top_words = []
    for topic in ldatopics:
        word_in_topic = []
        for wid in range(num_words):
            word_in_topic.append(topic[0][wid][1])
        top_words.append(word_in_topic)

    #### top_words == ldatopics

    # Create a topic-document matrix:
    d = ldamodel.get_document_topics(corpus)
    topic_doc_matrix = np.zeros((k_topics, len(d)), dtype=np.float16)
    topic_x_doc = np.zeros((1, len(d)), dtype=np.int8)
    list_topic_x_doc=[]
    for n,doc in enumerate(d):
        aux = np.reshape(doc, (len(doc), 2))
        topics = aux[:, 0]
        list_topic_x_doc.append(topics)
        if len(topics) > 0:
            topic_x_doc[0, n] = len(topics)
            for i in topics:
                topic_doc_matrix[int(i), n] = aux[int(np.nonzero(aux == int(i))[0])][1]

    all_keywords = []
    all_weigths = []
    idnewspaper=0
    for n in range(len(X_train_raw)):
        tokens_in_file = tokenize(X_train_raw[n], deacc=False)
        aux=[]
        for word in tokens_in_file:
            aux.append(word)

        aux = Dictionary([aux])
        word2id_in_file = aux.token2id

        topic_filewords_matrix = np.zeros((topic_x_doc[0,n], len(word2id_in_file)), dtype=float)

        for k_top in range(topic_x_doc[0,n]):
            topic = int(list_topic_x_doc[n][k_top])
            for word in word2id_in_file:
                if len(word) > 1 and word not in stopwords and word not in my_punctuation and word in word2id:
                    topic_filewords_matrix[k_top, word2id_in_file[word]] += topic_word_matrix[topic, word2id[word]]
                    # topic_sentence_matrix[k_top, s] += topic_doc_matrix[topic, n] * topic_word_matrix[topic, word2id[word]]

            topic_filewords_matrix[k_top,:] *= topic_doc_matrix[topic, n]

        # Suma por columnas para tener el peso acumulado de una palabra en todos los topics del documento
        weight_words = np.sum(topic_filewords_matrix, axis=0)
        doc_keywords, keyword_weight = get_keywords(n_keywords, weight_words, word2id_in_file)
        all_keywords.append(doc_keywords)
        all_weigths.append(keyword_weight/np.sum(keyword_weight))  # Normalizo pesos

        if ismain:
            if n == news_index[idnewspaper]:
                save_obj(all_keywords, 'data/%s.keyw.pkl' % news_name[idnewspaper])
                save_obj(all_weigths, 'data/%s.weight.pkl' % news_name[idnewspaper])
                all_keywords = []
                all_weigths = []
                idnewspaper+=1

    if not ismain:
        return all_keywords, all_weigths, word2id, id2word.id2token


def extract_articles_from_file(textpath, mynewspaper):
    print('extract_articles_from_file')

    content = []
    content_idx = [0]
    names = []
    for base, _, files in os.walk(textpath):
        if base == textpath:
            for fname in sorted(files):
                f = fname.split('.')
                if re.match(mynewspaper, f[0]):
                    title = '.'.join(f[:-1])
                    title = title.replace('text', 'title')
                    names.append(title[:-5])
                    print(title)

                    with codecs.open(os.path.join(textpath, fname), encoding='utf-8') as file_pointer:
                        file_content = file_pointer.readlines()

                    for id, text_line in enumerate(file_content):
                        if text_line.strip() != '':
                            content.append(text_line)
                        else:
                            print("Line %d is empty!" % id)
                    file_pointer.close()
                    content_idx.append(content_idx[-1]+id-1)
                    print("%d lines" % id)
    return content, content_idx[1:], names


if __name__ == '__main__':

    config = {
        'input_text_path' : '/extra/scratch03/lcabello/3.text2lema/clean',
        'newspaper' : '201701',  # sys.argv[1]
    }

    newspaper_content, newspaper_index, newspaper_name = extract_articles_from_file(config['input_text_path'], config['newspaper'])
    print(newspaper_index)
    print(newspaper_name)

    init = time()
    compute_keywords(newspaper_content, ismain=True, news_index=newspaper_index, news_name=newspaper_name)  # word2idx, id2wordx are WITHOUT stopwords
    endt = time()
    print('Elapsed time: %.4f s\n' % (endt-init))
