#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
import nltk

from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import tokenize

sys.path.append('../tools')
from utils import doc_processing
from IPython import embed

# -- Finding out the optimal number of topics --

def build_texts(fname):
    with open(fname) as f:
        for line in f:
            line = line.replace('_', ' ')
            yield line.decode('utf-8')


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lda_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lda_list = []
    for num_topics in range(1, limit+1):
        print("Topic %d" % num_topics)
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lda_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        # cm = CoherenceModel(model=lm, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit+1)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lda_list, c_v

def evaluate_perplexity(dictionary, corpus, texts, limit):
    perplex = np.zeros((1, limit), dtype=np.float16)

    lda_list = []
    for num_topics in range(1, limit+1):
        print("Topic %d" % num_topics)
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lda_list.append(lm)

        perplex[0, num_topics-1] = lm.log_perplexity(corpus)


    # Show graph
    x = range(1, limit+1)
    plt.plot(x, perplex.T)
    plt.xlabel("k topics")
    plt.ylabel("Perplexity")
    plt.show()

    return lda_list, perplex

if __name__ == '__main__':

    documents = list(build_texts('201703.www.heraldo.es.text'))

    stopwordsFilePath = '../stopwords_es_ES_enh.txt'

    processed_docs, stopwords = doc_processing(documents, stopwordsFilePath, doc=True)
    dictionary = Dictionary(processed_docs)
    # dictionary.filter_extremes(no_below=30, no_above=0.5, keep_n=None)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]

    k_topics = 110  # int(len(documents)/2)

    # 1) Matrix topic coherence:
    [ldalist, c_v] = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=processed_docs, limit=k_topics)

    embed()
    ldatopics = ldalist[9].show_topics(formatted=False)

    # 2) Perplexity
    [ldalist, perplexity] = evaluate_perplexity(dictionary=dictionary, corpus=corpus, texts=processed_docs, limit=k_topics)

