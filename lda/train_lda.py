#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, warnings, codecs, bz2
import gensim
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric
from time import time

import logging

warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

#######################################################################################
CHARSET = "utf-8"
MY_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'

class TextCorpusUtf8(object):
    """Iterate over sentences from the our corpus"""

    def __init__(self, fname, stopwords_file_path):
        self.fname = fname
        with codecs.open(stopwords_file_path, 'r', CHARSET) as ff:
            self.stopwords = [line.strip() for line in ff]

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        fin = codecs.open(self.fname, 'r', CHARSET)
        while True:
            sentence = fin.readline()  # load just 1 line into RAM
            if sentence:
                self.processed_sentence = self._doc_processing(sentence)
                yield self.processed_sentence
            else:
                break

    def _doc_processing(self, document):
        doc = []
        words_in_sent = gensim.utils.tokenize(document, deacc=False)
        for word in words_in_sent:
            if len(word) > 1 and word not in self.stopwords and word not in MY_PUNCTUATION:
                doc.append(word)
        return doc

########################################################################################

if __name__ == '__main__':
    FILENAME = 'upto201709'
    fname = '../lda_model/%s.text.lema.txt' % FILENAME
    assert os.path.exists(fname)
    flog = fname.replace('txt','log')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # filename=flog, filemode='w', f
    stopwords_file_path = '../../stopwords_es_ES_enh.txt'

    if not os.path.exists('../lda_model/%s.lema.no_below_70docs.mm' % FILENAME):
        init = time()
        processed_docs = TextCorpusUtf8(fname, stopwords_file_path)
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=70, no_above=1.0, keep_n=None)
        dictionary.save('../lda_model/%s.lema.no_below_70docs.dict' % FILENAME)  # store the dictionary, for future reference
        print("We create a dictionary, an index of all unique values: %s" % type(dictionary))
        print('Vocabulary size: %d' % len(dictionary))
        # 2004285 -> sin filtrar / 43649 -> filtrar que aparezcan en +100 docs / 54580 -> filtrar que aparezcan en +70 docs
        sys.stdout.flush()

        # TRAIN
        # compile corpus (vectors number of times each elements appears)
        corpus = [dictionary.doc2bow(text) for text in processed_docs]
        print ("Then convert tokenized documents to vectors: %s" % type(corpus))
        gensim.corpora.MmCorpus.serialize('../lda_model/%s.lema.no_below_70docs.mm' % FILENAME, corpus)  # Store to disk
        print ("Save the vectorized TRAIN corpus as a .mm file")
        sys.stdout.flush()

        # HOLDOUT AND TEST
        ho_processed_docs = TextCorpusUtf8('/extra/scratch03/Language/Periodicos/textos/3.text2lema/201710.text.lema.txt', stopwords_file_path)
        ho_corpus = [dictionary.doc2bow(text) for text in ho_processed_docs]
        gensim.corpora.MmCorpus.serialize('../lda_model/201710.lema.mm', ho_corpus)  # Store to disk
        test_processed_docs = TextCorpusUtf8('/extra/scratch03/Language/Periodicos/textos/3.text2lema/201711.text.lema.txt', stopwords_file_path)
        test_corpus = [dictionary.doc2bow(text) for text in test_processed_docs]
        gensim.corpora.MmCorpus.serialize('../lda_model/201711.lema.mm', test_corpus)  # Store to disk
        print ("Save the vectorized HOLDOUT AND TEST corpus as a .mm file")
        sys.stdout.flush()

        # if not os.path.exists('../lda_model/%s.lema.no_below_70docs.tfidf' % FILENAME):
        #     # Transform Text with TF-IDF
        #     tfidf = gensim.models.TfidfModel(corpus, id2word=dictionary, normalize=True)  # step 1 -- initialize a model
        #     print("Built TF-IDF matrix: %s" % type(tfidf))
        #     tfidf.save('../lda_model/%s.lema.no_below_70docs.tfidf' % FILENAME)

        endt = time()
        elapsed_time = endt-init
        print("Elapsed time: %.3f seconds\n\n" % elapsed_time)
        sys.stdout.flush()
        # exit the program to compress the .mm file SI gensim 2.2.0
        # $ bzip2 -z -v ../lda_model/upto201709.lema.70docs.mm  >> it replaces input file by its compression

    else:

        processed_docs = TextCorpusUtf8(fname, stopwords_file_path)

        if gensim.__version__ == '2.2.0':
            bz = bz2.BZ2File('../lda_model/%s.lema.no_below_70docs.mm.bz2' % FILENAME, mode='r')
            corpus = gensim.corpora.MmCorpus(bz)  # use this if you compressed the TFIDF output
            holdout_corpus = gensim.corpora.MmCorpus('../lda_model/201710.lema.mm')
            test_corpus = gensim.corpora.MmCorpus('../lda_model/201711.lema.mm')
            print("We load our compressed vector corpus: %s " % type(corpus))
        elif gensim.__version__ == '3.2.0':
            corpus = gensim.corpora.MmCorpus('../lda_model/%s.lema.no_below_70docs.mm' % FILENAME)  # use this if you compressed the TFIDF output
            holdout_corpus = gensim.corpora.MmCorpus('../lda_model/201710.lema.mm')
            test_corpus = gensim.corpora.MmCorpus('../lda_model/201711.lema.mm')
            print("We load our compressed vector corpus: %s" % type(corpus))
        else:
            print('Check gensim version')
            sys.exit()

        dictionary = gensim.corpora.Dictionary.load('../lda_model/%s.lema.no_below_70docs.dict' % FILENAME)
        print('Vocabulary size: %d' % len(dictionary))
        sys.stdout.flush()

        print("\nTopic modeling using LDA [...]")
        sys.stdout.flush()

        ## Define metrics ##
        '''
        Coherence: c_v gives the best results however it’s much slower than u_mass. It is due to the fact that c_v uses
        a sliding window for probability estimation and uses the indirect confirmation measure. The original sliding
        window moves over the document one word token per step: [a,b] [b,c] ... tweak the algorithm to run by skipping
        the already encountered tokens: [a,b] [c,d] so it throws an approximate result of c_v but it's faster
        '''
        pl_holdout = PerplexityMetric(corpus=holdout_corpus, logger="visdom", title="Perplexity (hold_out)")
        pl_test = PerplexityMetric(corpus=test_corpus, logger="visdom", title="Perplexity (test)")
        # pl_train = PerplexityMetric(corpus=corpus, logger="visdom", title="Perplexity (train)")
        ch_umass = CoherenceMetric(corpus=corpus, coherence="u_mass", logger="visdom", title="Coherence (u_mass)")
        ch_cv = CoherenceMetric(corpus=corpus, texts=processed_docs, coherence="c_v", logger="visdom", title="Coherence (c_v)")
        diff_kl = DiffMetric(distance="kullback_leibler", logger="visdom", title="Diff (kullback_leibler)")
        convergence_jc = ConvergenceMetric(distance="jaccard", logger="visdom", title="Convergence (jaccard)")

        callbacks = [pl_holdout, pl_test, ch_umass, ch_cv, diff_kl, convergence_jc]

        # k_topics = 100, alpha = 0.3, beta = 1
        k_topics = 50   # Elegido tras calcular coherence matrix
        alpha = 0.05    # 50.0/k_topics --- 1.0/k_topics
        eta = 1  # 0.01, 0.3
        niter = 200
        init = time()
        if os.path.exists('../lda_model/upto201709.no_below_70docs.%dk.iter_10.ldamodel' % k_topics):
            # lda = gensim.models.ldamodel.LdaModel(id2word=dictionary, num_topics=k_topics, alpha=alpha,
            #                                       eta=eta, iterations=niter, update_every=1, chunksize=10000,
            #                                       distributed=True, callbacks=callbacks)
            lda = gensim.models.ldamodel.LdaModel.load('../lda_model/upto201709.no_below_70docs.%dk.iter_10.ldamodel' % k_topics, mmap='r')
            lda.update(corpus=corpus, passes=5, update_every=1, eval_every=10, chunksize=10000,
                       iterations=niter, chunks_as_numpy=True, loaded=True)
        else:
            lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k_topics, alpha=alpha,
                                                  eta=eta, iterations=niter, update_every=1, chunksize=10000, passes=10,
                                                  distributed=True, callbacks=callbacks)
            '''
            2018-02-09 13:25:10,324 : INFO : running online (multi-pass) LDA training, X topics,
            10 passes over the supplied corpus of 850915 documents, updating model once every 40000 documents,
            evaluating perplexity every 400000 documents, iterating 200x with a convergence threshold of 0.001000
            '''

        lda.save('../lda_model/%s.no_below_70docs.%dk.ldamodel' % (FILENAME, k_topics))
        endt = time()
        elapsed_time = endt-init
        print("Elapsed time training the LDA model: %.3f seconds" % elapsed_time)
        # 6452.308 seconds - passes=1
