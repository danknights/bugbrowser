import re
import json
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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


def replace_all(doc, replacements):
    for i, j in replacements.iteritems():
        doc = doc.replace(i, j)
    return doc

# [a-zA-Z]{4,}|[0-9]+\t|\t|\n"
def split_iter(string, exp):
    return (x.group(0) for x in re.finditer(exp, string))


lower_transform = str.lower
test ='123456\tthese are words from the title\tthese are words from the document\n123456\tthese are words from the second title\tthese are words from the second document\n'
print(test)
print('*********')
for doc in split_iter(test, r'[\n]'):
    for group in split_iter(doc, r'[\t]'):

corpus_transforms = [lower_transform]

def split(orig, idname, docname, lemma):
    with open(orig) as origf , open(idname, 'w') as idf, open(docname, 'w') as docf:
        for line in origf:
            id, title, abstract = line.split('\t')
            idf.write(id+'\n')

            title  = ''.join(['T'+lemma(word.lower()) for word in split_iter(title, r'[a-zA-Z]{4,}')])

# document transform

# word transforms

# lemma_filter = WordNetLemmatizer().lemmatize

# abstracts = open('abstracts_titled_lower.txt').readlines()
# stop = set(stopwords.words('english'))

# l = WordNetLemmatizer()
# l = l.lemmatize
# bugs = Counter(json.loads(open('bug_frequency.json').read()))
# kept_bugs = [bug for bug in bugs if bugs[bug] >= 10]
# r = make_replacements(kept_bugs)
#
# with open('abstracts_cleaned.txt', 'w') as out:
#     for line in tqdm(open('abstracts_titled_lower.txt').readlines()):
#         line = replace_all(line, r)
#         id, title, abstract = line.strip().split('\t')
#         out.write('\t'.join([id, ' '.join(get_words(title, stop, l)), ' '.join(get_words(abstract, stop, l))]) + '\n')
