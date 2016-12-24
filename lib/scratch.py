import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
from collections import Counter
import json


def get_words(s, stop, lemma):
    return [str(lemma(x)) for x in re.split('[^a-zA-Z]+', s) if (len(x) > 3 and x not in stop)]

def make_replacements(bugs):
    replacements = {}
    for bug in bugs:
        genus, species = bug.split()
        uid = str(''.join((genus, species)))
        replacements[str(bug)] = uid
        replacements[str("{}. {}".format(genus[0], species))] = uid
    return replacements

abstracts = open('abstracts_titled_lower.txt').readlines()
stop = set(stopwords.words('english'))

l = WordNetLemmatizer()
l = l.lemmatize
bugs = Counter(json.loads(open('bug_frequency.json').read()))
kept_bugs = [bug for bug in bugs if bugs[bug]>=10]
r = make_replacements(kept_bugs)
    
with open('abstracts_cleaned.txt', 'w') as out:
    for line in tqdm(open('abstracts_titled_lower.txt').readlines()):
        line = replace_all(line, r)
        id, title, abstract = line.strip().split('\t')
        out.write('\t'.join([id, ' '.join(get_words(title, stop, l)), ' '.join(get_words(abstract, stop, l))])+'\n')
