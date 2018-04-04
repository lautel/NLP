# _*_ coding: utf-8 _*_
"""

@author: Laura Cabello

date: 31-08-2017
"""
from __future__ import print_function
from gensim.utils import tokenize
from nltk import sent_tokenize
import codecs, os, re, pickle
import numpy as np
from time import time

from extract_keywords import compute_keywords

###############################################################################
#  Util functions
###############################################################################

class Article(object):
    def __init__(self, identity):
        self.identity = identity
        self.title = ''
        self.content = ''

    def __str__(self):
        print_string = 'Identity: ' + str(self.identity) + '\n' + 'Title: ' + self.title + '\n' + 'Content: \n' + self.content
        return print_string.encode('utf8')

    def set_title(self, title):
        self.title = title

    def set_content(self, content):
        self.content = content


class Word(object):
    def __init__(self, id):
        self.id = id
        self.word_string = ''
        self.vector = []

    def set_word_settings(self, word_string, vector):
        self.word_string = word_string
        self.vector = vector

class Word2Vec(object):
    def __init__(self, w2id, id2w):
        self.w2id = w2id
        self.id2w = id2w
        self.word_info = list(range(len(self.w2id)))
        self.scale = 0

    def open_file(self, filename):
        with open(filename) as ff:
            for line in ff:
                word_string = line.split(' ')[0]
                try:
                    word_string = word_string.decode('utf-8')
                except UnicodeDecodeError:
                    print("UnicodeDecodeError in %s" % word_string)
                    continue

                if word_string in self.w2id:
                    str_vector = np.asarray(line.split(' ')[1:])
                    if str_vector[-1] == '\n':
                        str_vector = str_vector[:-1]
                    if len(str_vector) == 300:
                         # self.vector = str_vector.astype(np.float)  ## Mucho más lento que la linea de abajo
                        vector = [float(i) for i in str_vector]
                        word = Word(self.w2id[word_string])
                        word.set_word_settings(word_string, vector)
                        self.word_info[self.w2id[word_string]] = word
                        self.scale += np.asarray(vector).std() * np.sqrt(12) / 2  # uniform and not normal
                    else:
                        print("The word %s has a str_vector less than 300 dim" % word_string)
                else:
                    pass
            self.scale = self.scale/len(self.word_info)

    def compute_doc2vec_by_sentence(self, newspaper, list_of_keywords, list_of_w, embed_dim=100):
        article_vector = []
        for ii, article in enumerate(newspaper):

            # Sentences in article to Vector
            sent_in_article = sent_tokenize(article.content)
            sentence_vector = np.zeros((embed_dim, len(sent_in_article)), dtype='float32')

            list_of_keywords_ii = list(tokenize(list_of_keywords[ii], deacc=False))
            keyw2id, id2keyw = generate_dict(list_of_keywords_ii)

            for s, sentence in enumerate(sent_in_article):  # must set spanish.pickle en __init__
                words_in_sent = list(tokenize(sentence, deacc=False))
                word_weight = (np.min(list_of_w[ii])/2) * np.ones((len(words_in_sent),))  ## >> Así, o iniciandolos directamente todos a 1
                word2vec = np.zeros((embed_dim, len(words_in_sent)), dtype='float32')

                for w, word in enumerate(words_in_sent):

                    if word in list(tokenize(list_of_keywords[ii], deacc=False)):  # implica que no es una stopword
                        try:
                            word2vec[:, w] = self.word_info[self.w2id[word]].vector
                        except AttributeError:
                            # palabra que está en vocabulario de la noticia pero no en glove file (stopwords y asi)
                            word2vec[:, w] = np.random.uniform(low=-self.scale, high=self.scale, size=embed_dim)
                        if word in list_of_keywords_ii:
                            word_weight[w] = list_of_w[ii][keyw2id[word]]
                sentence_vector[:,s] = np.sum(( np.tile(word_weight, (embed_dim,1)) * word2vec), axis=1)  # --> suma ponderada, vector unitario
                sentence_vector[:,s] = sentence_vector[:,s] / np.sum(sentence_vector[:,s])  # módulo 1

            # Article to Vector
            article_vector.append(np.product(sentence_vector, axis=1))  # o producto vectorial: np.cross(a, b)
        return article_vector

    def compute_doc2vec(self, newspaper, list_of_keywords, list_of_w, stopword, embed_dim=100):
        article_vector = []
        for ii, article in enumerate(newspaper):
            words_in_doc = list(tokenize(article.content, deacc=False))
            word_weight = (np.min(list_of_w[ii])/2) * np.ones((len(words_in_doc),))  # se conserva para las palabras que no son keyword ni stopword
            word2vec = np.zeros((embed_dim, len(words_in_doc)), dtype='float32')

            list_of_keywords_ii = list(tokenize(list_of_keywords[ii], deacc=False))
            keyw2id, id2keyw = generate_dict(list_of_keywords_ii)
            # cum_weights = 0
            for w, word in enumerate(words_in_doc):
                word = word.lower()
                try:
                    word2vec[:, w] = self.word_info[self.w2id[word]].vector
                except AttributeError:
                    # palabra que está en vocabulario de la noticia pero no en glove file (stopwords y asi)
                    word2vec[:, w] = np.random.uniform(low=-self.scale, high=self.scale, size=embed_dim)
                if word in list_of_keywords_ii:
                    word_weight[w] = list_of_w[ii][keyw2id[word]]
                elif word in stopword:
                    word_weight[w] = 0.001
                # cum_weights += word_weight[w]

            doc_vector = np.sum((word_weight * word2vec), axis=1)
            doc_vector = doc_vector / np.linalg.norm(doc_vector)
            if np.isnan(doc_vector).any():
                print(article.title)
                doc_vector = np.sum((word_weight * word2vec), axis=1)

            # Article to Vector
            article_vector.append(doc_vector)  # o producto vectorial: np.cross(a, b)
        return np.asarray(article_vector)


def get_parameters():
    parameter = dict()
    parameter['word2vec_file'] = '../WordRepresentation/all.lema.300.min100.vectors.txt'  # all.Glove.lema.vectors.100.010.txt  # all.lema.300.min100.vectors.txt
    parameter['input_text_path'] = '/extra/scratch03/lcabello/3.text2lema/clean'  # '../3.text2lema/clean'
    parameter['input_title_path'] = '/extra/scratch03/lcabello/2.docsent2title'  # '../2.docsent2title'
    parameter['output_text_path'] = 'results/'

    parameter['newspaper'] = '201701'  # sys.argv[1]
    parameter['ntimewordappearance'] = 1  # Ahora quizá no tenga mucho sentido contar un numero porque no cargamos todos
    #  los periodicos de vez, sino uno a uno y los nombres propios puede que no aparezcan muchas veces. Por ello, con
    # tal de que la palabra tenga representación en word2vec la damos por valida.
    return parameter


def get_word_count_from_text(content):
    # print('Word count')
    word_counts = dict()
    unk_word = '<unk>'.decode('utf-8')
    word_counts[unk_word] = 0

    for article_index in xrange(len(content)):
        words_in_sent = list(tokenize(content[article_index].content, deacc=False))
        words_in_title = list(tokenize(content[article_index].title, deacc=False))
        for word in words_in_sent:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        for word in words_in_title:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    return word_counts


def generate_dict_with_word_appearance_more_than(word_counts, low_limit_appearance):
    word_to_index = dict()
    index_to_word = dict()
    word_index = 0
    for word in word_counts:
        if word_counts[word] >= low_limit_appearance:
            word_to_index[word] = word_index
            index_to_word[word_index] = word
            word_index += 1
    return word_to_index, index_to_word


def generate_dict(content):
    word_to_index = dict()
    index_to_word = dict()
    word_index = 0

    for article_index in xrange(len(content)):
        try:
            words_in_sent = list(tokenize(content[article_index].content, deacc=False))
        except:
            words_in_sent = list(tokenize(content[article_index], deacc=False))
            pass

        for word in words_in_sent:
            word = word.lower()
            if word not in word_to_index:
                word_to_index[word] = word_index
                index_to_word[word_index] = word
                word_index += 1
            else:
                continue
    return word_to_index, index_to_word


def extract_articles_from_file(text_file, title_file):
    print('extract_articles_from_file')
    with codecs.open(text_file, encoding='utf-8') as file_pointer:
        file_content = file_pointer.readlines()
    with codecs.open(title_file, encoding='utf-8') as file_pointer2:
        file_title = list(file_pointer2.readlines())

    content = []
    for id, text_line in enumerate(file_content):
        article = Article(id)
        if text_line.strip() != '':
            words_in_text_line = text_line.split(' ')
            article.set_content(' '.join(words_in_text_line))
            article.set_title(file_title[id].replace('_', ' '))
            content.append(article)
        else:
            print("Line %d is empty!" % id)
    file_pointer.close()
    file_pointer2.close()

    return content


def save_obj(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=2)


def load_obj(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)


###############################################################################
###############################################################################

if __name__ == '__main__':

    config = get_parameters()
    doc2vec = []

    file = codecs.open('../stopwords_es_ES_enh.txt', 'r', 'utf-8')
    stopwords = [line.strip() for line in file]
    file.close()

    contador = 0
    for base, _, files in os.walk(config['input_text_path']):
        if base == config['input_text_path']:
            for fname in sorted(files):
                f = fname.split('.')
                if re.match(str(config['newspaper']), f[0]):
                    title = '.'.join(f[:-1])
                    title = title.replace('text', 'title')
                    print(title)

                    newspaper_content = extract_articles_from_file(os.path.join(base, fname), os.path.join(config['input_title_path'], title))

                    init = time()
                    # [list_of_keywords, list_of_weights, _, _] = compute_keywords(newspaper_content)  # word2idx, id2wordx SON SIN stopwords
                    # save_obj(list_of_keywords, 'data/%s.keyw.pkl' % '.'.join(f[:-2]))
                    # save_obj(list_of_weights, 'data/%s.weight.pkl' % '.'.join(f[:-2]))
                    list_of_keywords = load_obj('data/%s_keyw.pkl' % '.'.join(f[:-2]))
                    list_of_weights = load_obj('data/%s_weight.pkl' % '.'.join(f[:-2]))

                    # word_count = get_word_count_from_text(newspaper_content)
                    [w2idx, idx2w] = generate_dict(newspaper_content)  # Aquí sí cuentan todas las palabras

                    w2v = Word2Vec(w2idx, idx2w)
                    w2v.open_file(config['word2vec_file'])

                    doc2vec.append(w2v.compute_doc2vec(newspaper_content, list_of_keywords, list_of_weights, stopwords, embed_dim=300))
                    endt = time()
                    print('Elapsed time: %.4f s\n' % (endt-init))

            np.savetxt(fname='results/_201701.doc2vec.noStopw.normal.vectors', X=np.concatenate(doc2vec), fmt='%.10e')
            print('Fin')
            break
