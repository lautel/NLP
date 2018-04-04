# NLP

Content:

* **doc2vec/**   word embeddings applied to entire documents (news from Spanish newspapers) in order to get a single embedding from each input document (news). It is based on keyword extraction to weight words. The image .png shows an example of results obtained with news collected on January-2017 from an online Spanish newspaper. doc2vec.py is the main script.

* **lda/**   LDA model training making use of gensim. The input corpus is iteratively loaded and the preprocessed. Different measures are defined in order to study the training process and ensure convergence, as shown in the .png image. Training is distributed, using the power of 4 CPUs. 

* **text2picto/**   text2picto.py is the main script that convert a given text to pictograms [some code snippets have been removed]. It exploits different NLP techniques such as word embeddings (trained earlier with word2vec) and LDA model to assign topics to documents and weight words according to it. The script uses functions defined in eval_word_neighbours.py like *eval* to retrieve the nearest neighbours from a word, and *eval_with_context_lda* to get a sentence embedding and compare it to different sentences embedding (useful if dealing with polysemy). 
 google_api.py sends queries Google searching for images.  request_api.py sends queries to our pictogram database to retrieve a pictogram.

* **tools/**   a small set of util functions. 


