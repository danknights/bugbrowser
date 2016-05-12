"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying Non-negative Matrix Factorization
and Latent Dirichlet Allocation on a corpus of documents and
extract additive models of the topic structure of the corpus.
The output is a list of topics, each represented as a list of terms
(weights are not shown).

The default parameters (n_samples / n_features / n_topics) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).
"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from nltk.corpus import stopwords
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
    print()

def get_last_col(fname, delimiter='\t'):
    with open(fname, 'r') as f:
        for line in f:
            try:
               yield line.split(delimiter, 2)[2]
            except IndexError:
               continue

print("Loading dataset...")
vectorizer = CountVectorizer(stop_words='english')
lda = LatentDirichletAllocation(n_topics=10, batch_size=10000, verbose=1, n_jobs=-1)
print("Performing LDA...")
pipe = Pipeline([('vectorizer', vectorizer), ('lda', lda)])
pipe.fit(get_last_col('doc/abstracts_with_titles.txt'))
print("\nTopics in LDA model:")
feature_names = vectorizer.get_feature_names()
print_top_words(lda, feature_names, 10)
print("Saving model...")
joblib.dump(lda, 'lda.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

