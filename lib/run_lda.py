from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import models
from gensim.corpora import dictionary, mmcorpus
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from argparse import ArgumentParser
import logging
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# python run_lda.py -u -b ../doc/bug_list.txt -f ../doc/abstracts/faecalibacterium\ prausnitzii.pmed -d practice.dict -c practice.corpus -m practice -t 10 -r ../doc/replacements.json
def replace_all(doc, replacements):
    for i, j in replacements.iteritems():
        doc = doc.replace(i, j)
    return doc

def get_words(id_abstract, bug_map, tokenizer, lemmatizer, stemmer, stop):
    '''pulls out the abstract text, converts it to unicode, removes accents, and converts to lower case, returns as a list of tokens'''
    id_abstract = replace_all(id_abstract.lower(), replacements)
    title = id_abstract.split('\t')[1].split(' ')
    abstract = id_abstract.split('\t')[2].split(' ')
    boosted_title_abstract  = ' '.join(title*20+abstract)
    return [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(boosted_title_abstract) if word.lower() not in stop]

def generateReplacements(bugf_name):
    label='bug'
    bug_id = 1
    replacements = {}
    with open(bugf_name) as bugf:
        for line in bugf:
            split = line.lower().strip().split()
            if len(split) == 2:
                unique_id = label+str(bug_id)
                genus, species = split
                replacements[genus+" "+species] = unique_id
                replacements[genus[0]+". "+species] = unique_id
                bug_id += 1
    return replacements
    
parser = ArgumentParser(description='Interface to update model.')
parser.add_argument('-u', '--update', action='store_true', help='perform update with supplied filenames')
parser.add_argument('-f', '--file', help='name of file containing abstracts.')
parser.add_argument('-d', '--dictionary', help='name of file containing dictionary')
parser.add_argument('-m', '--model', help='name of file containing model')
parser.add_argument('-c', '--corpus', help='name of file containing corpus')
parser.add_argument('-t', '--topics', help='number of topics')
parser.add_argument('-a', '--alpha', help='value of alpha hyperparameter')
parser.add_argument('-e', '--eta', help='value of eta hyperparameter')
parser.add_argument('-b', '--bugs', help='file containing list of bugs')
parser.add_argument('-r', '--replacements', help='file containing id<->genus species map')
args = parser.parse_args()
args = vars(args)

if args['update']:
    stemmer, tokenizer, lemmatizer = PorterStemmer(), RegexpTokenizer(r'[a-zA-Z0-9]{3,}'), WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    replacements = generateReplacements(args['bugs'])
    with open(args['replacements'], 'w') as replacef:
        json.dump(replacements, replacef)
    with open(args['file']) as abstractf:
        word_dict = dictionary.Dictionary(
            get_words(id_abstract, replacements, tokenizer, lemmatizer, stemmer, stop) for id_abstract in abstractf)
        word_dict.save(args['dictionary'])
    with open(args['file']) as abstractf:
        corpus = (word_dict.doc2bow(get_words(id_abstract, replacements, tokenizer, lemmatizer, stemmer, stop)) for id_abstract in
                  abstractf)
        mmcorpus.MmCorpus.serialize(args['corpus'], corpus)
word_dict = dictionary.Dictionary().load(args['dictionary'])
corpus = mmcorpus.MmCorpus(args['corpus'])
t = int(args['topics'])
e = args['eta']
if e == 'None':
    e = None
model = models.LdaMulticore(corpus, id2word=word_dict, num_topics=t, alpha=args['alpha'], eta=e, passes=4, workers=31)
model.save(args['model'])
