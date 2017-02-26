import json, logging
from collections import namedtuple
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import models
from gensim.corpora import dictionary, mmcorpus
from lib.make_clouds import make_clouds
from lib.helpers import replace_all, bugs_exceeding_count, get_replacements
from lib.process_model import create_json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Files = namedtuple('Files', ['bugs', 'frequency', 'tfidf', 'abstracts', 'replacements', 'corpus', 'model', 'docsXtopics'])

def run_lda(tfidfm, files, alpha, eta, num_topics, update):
    if update:
        corpus = mmcorpus.MmCorpus(files.corpus)
        model = models.LdaModel(tfidfm[corpus], id2word=tfidfm.id2word, num_topics=num_topics, alpha=alpha, eta=eta, passes=2)
        model.save(files.model)
        # save the documents X topics matrix of the model
        mmcorpus.MmCorpus.serialize(files.docsXtopics, model[corpus])
        return model
    return models.LdaModel.load(files.model)


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


def get_tfidf(files, update):
    if update:
        tokenizer, lemmatizer, stop = RegexpTokenizer(r'[a-z]{4,}'), WordNetLemmatizer(), set(stopwords.words('english'))
        bug_list = bugs_exceeding_count(files, 10)
        replacements = get_replacements(bug_list)
        with open(files.replacements, 'w') as outf:
            json.dump(replacements, outf)
        with open(files.abstracts) as documents:
            word_dict = dictionary.Dictionary(
                get_words(id_abstract, replacements, tokenizer, lemmatizer) for id_abstract in documents)
            word_dict.filter_tokens(bad_ids=stop) # get rid of stop words
            word_dict.filter_extremes() # filter out
            documents.seek(0) # reset to beginning of file iterator
            corpus = (word_dict.doc2bow(get_words(id_abstract, replacements, tokenizer, lemmatizer)) for
                      id_abstract in documents)
            mmcorpus.MmCorpus.serialize(files.corpus, corpus)
        tfidf_model = models.tfidfmodel.TfidfModel(dictionary=word_dict, id2word=word_dict)
        tfidf_model.save(files.tfidf)
    tfidf_model = models.tfidfmodel.TfidfModel.load(files.tfidf)
    return tfidf_model

def print_results(lda_model):
    if lda_model is None:
        print("Model cannot be None.")
    for topicn, word_weight_list in lda_model.show_topics(num_topics=-1, num_words=20, formatted=False):
        print(' '.join(word for word, weight in word_weight_list))


files = Files('../doc/bug_list.txt', '../doc/bug_frequency.json', '../doc/tfidf_sample.tfidf', '../doc/abstract_samples.txt', \
              '../doc/replacement_sample.json','../doc/corpus.mm','../doc/sample_model.model', '../doc/docsXtopics.mm')

tfidf_model = get_tfidf(files, update=False)
lda_model = run_lda(tfidf_model, files, alpha=.01, eta=None, num_topics=50, update=False)
print_results(lda_model)
create_json(files, 10, 10, 5)
# make_clouds(files)
