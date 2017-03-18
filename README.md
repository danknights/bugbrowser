**BugBrowser**
=================

BugBrowser provides insight into the vast amount of literature regarding the human gut microbiome. It is a self-contained tool for scraping PubMed for abstracts, pipelining these abstracts to Gensim [1], performing topic-modeling to extracting per-topic and per-bacterium insight. 

## Overview
`pubmed_query.py` is responsible for gathering all abstracts for all bacteria of interest via Pubmed’s API. 

`run_lda.py` is responsible for performing the topic-modeling using the open source library Gensim. Several simplifying actions are taken:
* Only alphabetic characters are retained.
* Words of fewer than 4 characters are removed.
* Lemmatization is performed if possible.
* Common english stop-words are removed.
* Bacteria with fewer than 10 abstracts on Pubmed lack sufficient data for meaningful insight via topic-modeling, and are excluded.
* Abstract titles are treated as more meaningful than body text by increasing the occurence of words in the title.

`make_clouds.py` will generate word clouds for the topics of a topic model.


`create_json.py` will perform post-processing, taking the documents X words and topics X words matrices and extracting the top topics per bacterium, the top documents mentioning the bacterium by proportion of the topic, and these top documents filtered by each word in each of the top topics. 

##Citations
[1] Rehurek,R., Sojka,P. (2010) Software Framework for Topic Modelling with Large Corpora, Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, 45-50.

