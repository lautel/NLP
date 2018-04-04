#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import sys, argparse
import json
from time import time

from utils import *
from request_api import RequestAPI, ImageScraper
from google_api import RequestGoogle
from eval_word_neighbours import EvalWordNeighbours

warnings.filterwarnings("ignore")


class Text2Picto:
    def __init__(self, stopwordPath='assets/stopwords_es_ES_PICTOS.txt',
                 postag_file='assets/upto201709.text.lema.tagged.200.wordcount.txt',
                 dictionary_pictos_file='assets/diccionario_pictogramas.txt', palabrasPath='assets/palabras.txt',
                 embedding_file='assets/upto201709.text.lema.tagged.200.vectors.txt', picto_directory='results-pictosummary/'):

        t1 = time()
        self.picto_directory = picto_directory
        self.stopwords = open_stopwords(stopwordPath)
        self.postag_dictionary = open_postag_dictionary(postag_file)
        self.dictionary = open_dictionary(dictionary_pictos_file, palabrasPath)

        self.client = RequestAPI(language='ES',
                            catalog='colorpictos',  # bwpictos para pictogramas de Blanco y Negro
                            thumbnailsize=150,  # Tamaño de la minuatura de la que quiero me genere URL
                            )
        self.google = RequestGoogle()
        self.scraper = ImageScraper(download_path=picto_directory)
        self.word2vec = EvalWordNeighbours(embedding_file, postag_file)

        t2 = time()
        print("%.3f seconds elapsed loading resources" % (t2-t1))
        sys.stdout.flush()

    def __call__(self, use_lda, input_files):
        t1 = time()
        for text_path in input_files:
            assert text_path.split('.')[-1] == 'json'
            NAME = os.path.basename(text_path).split('.')[0]
            print("\nFile : %s\n" % NAME)
            outf = codecs.open('results-pictosummary/output.%s.LDA.%s.txt' % (NAME,use_lda), 'w', 'utf-8')
            action = False
            acc=[]
            lines = list(codecs.open(text_path, "r", "utf-8"))
            data = json.loads(lines[0])  # json de una unica 'linea' siempre
            data = data['article']
            for s_id, sentence_object in enumerate(data):
                '''
                code removed 
                '''
                for tuple_data in enumerate(zip(words,postag,original)):
                  '''
                  code removed 
                  '''
                compose_text_2_picture(pictogramas, s_id, NAME, size=(256,256))
                if pictoid:
                    acc.append(test_pictoid(pictogramas, true_pictoid, outf))

            print("TOTAL ACCURACY IN {} SENTENCES: {}%".format(len(acc), np.mean(acc)))
            outf.write("\nTOTAL ACCURACY IN {} SENTENCES: {}%".format(len(acc), np.mean(acc)))

            t2 = time()
            print("%.3f seconds elapsed in file %s" % (t2-t1, NAME))
            outf.close()


if __name__ == '__main__':
    '''
    ejemplo de uso para 2 documentos:
    python text2picto.py -lda 1 -in summaries/test.json summaries/ej_polisemia.json

    NOTA: json de entrada debe contener el documento original ya resumido. Los campos del json son: frase original,
    frase lematizada y POS tag de cada palabra.

    NOTA: para escribir el html final hay que escribir la ruta completa a esta carpeta en la línea 151 de este script.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-lda', dest='lda', type=int, default=1, help='activado a 1 por defecto. Poner a 0'
                                                                                 'para no utilizar LDA')
    parser.add_argument('-in', dest='input', type=str, nargs='+', default=['summaries/frases_cuentos_final.json'], help='input json files')
    args = parser.parse_args()
    assert len(args.input) > 0
    if args.lda == 0:
        use_lda = False
    elif args.lda == 1:
        use_lda = True
    else:
        print("Wrong value for -lda variable")
        sys.exit()

    txt2pic = Text2Picto()
    txt2pic(use_lda, args.input)
