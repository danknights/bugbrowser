from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import models
from gensim.corpora import dictionary, mmcorpus
from nltk.corpus import stopwords
from argparse import ArgumentParser
import logging, json, os
from make_clouds import make_clouds
from process_model import create_json


def main():
    # configure logging for gensim
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # parse command line parameters
    args = getArgs()
    # extract file paths and parameters from command line arguments
    corpusf, dictf, docsf, replacementf, model_dir, bugf, modelf, docsXtopicsf, base_model_name = getFilePaths(args)
    docn, eta, alpha, num_topics, topicn, wordn = get_parameters(args)
    # set up corpus and dictionary, create if necessary
    corpus, word_dict = get_corpus_and_dict(bugf, corpusf, dictf, docsf, replacementf, args['update'])
    # run LDA on the corpus, save model
    run_lda(alpha, corpus, docsXtopicsf, eta, modelf, num_topics, word_dict)
    # create word cloud for each topic
    make_clouds(modelf, base_model_name, replacementf, wordn)
    # create json files for each bug for use in browser
    create_json(base_model_name, modelf, replacementf, docsXtopicsf, docsf, bugf, corpusf, wordn, topicn, docn)


def run_lda(alpha, corpus, docsXtopicsf, eta, modelf, num_topics, word_dict):
    model = models.LdaModel(corpus, id2word=word_dict, num_topics=num_topics, alpha=alpha, eta=eta,
                                passes=2)
    model.save(modelf)
    # save the documents X topics matrix of the model
    mmcorpus.MmCorpus.serialize(docsXtopicsf, model[corpus])


# python run_lda.py -u -b ../doc/bug_list.txt -f ../doc/abstracts/faecalibacterium\ prausnitzii.pmed -d practice.dict -c practice.corpus -m practice -t 10 -r ../doc/replacements.json
def replace_all(doc, replacements):
    for i, j in replacements.iteritems():
        doc = doc.replace(i, j)
    return doc


def get_words(id_abstract, replacements, tokenizer, lemmatizer, stemmer, stop):
    '''pulls out the abstract text, converts it to unicode, removes accents, and converts to lower case, returns as a list of tokens'''
    id_abstract = replace_all(id_abstract.lower(), replacements)
    title = id_abstract.split('\t')[1].split(' ')
    abstract = id_abstract.split('\t')[2].split(' ')
    boosted_title_abstract = ' '.join(title * 20 + abstract)
    return [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(boosted_title_abstract) if
            word.lower() not in stop]


def get_replacements(bugf_name):
    label = 'bug'
    bug_id = 1
    replacements = {}
    with open(bugf_name) as bugf:
        for line in bugf:
            split = line.lower().strip().split()
            if len(split) == 2:
                unique_id = label + str(bug_id)
                genus, species = split
                replacements[genus + " " + species] = unique_id
                replacements[genus[0] + ". " + species] = unique_id
                bug_id += 1
    return replacements


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


def get_corpus_and_dict(bugf, corpusf, dictf, docsf, replacementf, update):
    if update:
        stemmer, tokenizer, lemmatizer, stop = PorterStemmer(), RegexpTokenizer(r'[a-zA-Z0-9]{3,}'), \
                                               WordNetLemmatizer(), set(stopwords.words('english'))
        replacements = get_replacements(bugf)
        with open(replacementf, 'w') as outf:
            json.dump(replacements, outf)
        with open(docsf) as documents:
            word_dict = dictionary.Dictionary(
                get_words(id_abstract, replacements, tokenizer, lemmatizer, stemmer, stop) for id_abstract in documents)
            word_dict.save(dictf)
            documents.seek(0)
            corpus = (word_dict.doc2bow(get_words(id_abstract, replacements, tokenizer, lemmatizer, stemmer, stop)) for
                      id_abstract in documents)
            mmcorpus.MmCorpus.serialize(corpusf, corpus)
    word_dict = dictionary.Dictionary().load(dictf)
    corpus = mmcorpus.MmCorpus(corpusf)
    return corpus, word_dict


def getFilePaths(args):
    base_model_name = '_'.join((args['topics'], args['alpha'], args['eta']))
    model_dir = '../doc/models/' + base_model_name + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    doc_dir = '../doc/'
    docsf = doc_dir+args['file']
    base_docsf = args['file'].split('.')[0]
    dictf = doc_dir + base_docsf + '.dict'
    corpusf = doc_dir + base_docsf + '.corpus'
    replacementf = doc_dir + 'replacements.json'
    bugf = doc_dir+args['bugs']
    modelf = model_dir+base_model_name
    docsXtopicsf = model_dir+'docsXtopics.corpus'
    return corpusf, dictf, docsf, replacementf, model_dir, bugf, modelf, docsXtopicsf, base_model_name


def getArgs():
    parser = ArgumentParser(description='Interface to run LDA on abstracts')
    parser.add_argument('-u', '--update', action='store_true', help='perform update with supplied filenames')
    parser.add_argument('-f', '--file', default='abstracts_with_titles.txt', help='name of file containing abstracts.')
    parser.add_argument('-t', '--topics', help='number of topics')
    parser.add_argument('-a', '--alpha', default='symmetric', help='value of alpha hyperparameter')
    parser.add_argument('-e', '--eta', default='None', help='value of eta hyperparameter')
    parser.add_argument('-b', '--bugs', default='bug_list.txt', help='file containing list of bugs')
    parser.add_argument('-wc', '--word count', default='30', help='number of words in word cloud')
    parser.add_argument('-dc', '--doc count', default='5', help='number of related documents per term to store')
    parser.add_argument('-tc', '--topic count', default='5', help='number of topics per bug to store')
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == '__main__':
    main()
