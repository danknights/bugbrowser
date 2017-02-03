import logging, json, os
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import models
from gensim.corpora import dictionary, mmcorpus
from argparse import ArgumentParser
from make_clouds import make_clouds
from helpers import replace_all, bugs_exceeding_count, get_replacements


def run_lda(alpha, tfidfm, docsf, corpusf, docsXtopicsf, eta, modelf, num_topics, update):
    if update:
        corpus = mmcorpus.MmCorpus(corpusf)
        model = models.LdaModel(tfidfm[corpus], id2word=tfidfm.id2word, num_topics=num_topics, alpha=alpha, eta=eta, passes=2)
        model.save(modelf)
        # save the documents X topics matrix of the model
        mmcorpus.MmCorpus.serialize(docsXtopicsf, model[corpus])
        return model
    return models.LdaModel.load(modelf)


# python run_lda.py -u -b ../doc/bug_list.txt -f ../doc/abstracts/faecalibacterium\ prausnitzii.pmed -d practice.dict -c practice.corpus -m practice -t 10 -r ../doc/replacements.json


def get_words(line, replacements, tokenizer, lemmatizer):
    '''pulls out the abstract text, converts it to unicode, removes accents, and converts to lower case, returns as a list of tokens'''
    line = line.lower()
    line = replace_all(line, replacements)
    _, title, abstract = line.split('\t')
    titlewords = title.split(' ')
    boosted_title_abstract = ' '.join(titlewords * 5) + abstract # make words in title appear more frequently

    return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(boosted_title_abstract)]


def get_parameters(args):
    num_topics = int(args['topics'])
    topicn = int(args['topic count'])
    wordn = int(args['word count'])
    docn = int(args['doc count'])
    eta = args['eta']
    if eta == 'None':
        eta = None
    alpha = args['alpha']
    return docn, eta, alpha, num_topics, topicn, wordn


def get_tfidf(bugf, bugfreqf, tfidff, docsf, replacementf, corpusf, update):
    if update:
        tokenizer, lemmatizer, stop = RegexpTokenizer(r'[a-z]{4,}'), WordNetLemmatizer(), set(stopwords.words('english'))
        bug_list = bugs_exceeding_count(bugf, bugfreqf, 10)
        replacements = get_replacements(bug_list)
        with open(replacementf, 'w') as outf:
            json.dump(replacements, outf)
        with open(docsf) as documents:
            word_dict = dictionary.Dictionary(
                get_words(id_abstract, replacements, tokenizer, lemmatizer) for id_abstract in documents)
            word_dict.filter_tokens(bad_ids=stop) #get rid of stop words
            word_dict.filter_extremes() #filter out
            documents.seek(0)
            corpus = (word_dict.doc2bow(get_words(id_abstract, replacements, tokenizer, lemmatizer)) for
                      id_abstract in documents)
            mmcorpus.MmCorpus.serialize(corpusf, corpus)
        tfidf_model = models.tfidfmodel.TfidfModel(dictionary=word_dict, id2word=word_dict)
        tfidf_model.save(tfidff)
    tfidf_model = models.tfidfmodel.TfidfModel.load(tfidff)
    return tfidf_model


tfidf_model = get_tfidf('../doc/bug_list.txt', '../doc/bug_frequency.json', '../doc/tfidf_sample.tfidf', '../doc/abstract_samples.txt', '../doc/replacement_sample.json','../doc/corpus.mm', False)
lda_model = run_lda(.01, tfidf_model, '../doc/abstract_samples.txt', '../doc/corpus.mm', '../doc/docsXtopics.mm', None, '../doc/sample_model.model', 100, False)