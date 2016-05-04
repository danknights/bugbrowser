import os, sys, json
import numpy as np
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
from sklearn.preprocessing import normalize


def makeclouds(modelf, base_model_name, replacementsf, n_words):
    # set locations
    output_d = '../browser/clouds/' + base_model_name + '/'
    if not os.path.exists(output_d):
        os.makedirs(output_d)
    # create wordcloud generator
    wc = WordCloud(width=1000, height=500, background_color='white')

    print 'Loading model'
    model = LdaModel.load(modelf)
    beta = model.expElogbeta

    print 'Normalizing by topics, and by words'
    pTW = normalize(beta, axis=0)
    pWT = normalize(beta, axis=1)

    # load bug<->id map, then invert to id<-> bug
    bug_to_id = json.loads(open(replacementsf).read())
    id_to_bug = {v: k for k, v in bug_to_id.items() if "." not in k}

    for i in range(len(beta)):
        # compute RAR
        t_rar = np.sqrt(pTW[i] * pWT[i])
        top_word_ids = t_rar.argsort()[:-1 - n_words:-1]
        top_words = [model.id2word.id2token[wordid] for wordid in top_word_ids]
        top_words = [id_to_bug[word] if word in id_to_bug else word for word in top_words]
        wc.fit_words(zip(top_words, t_rar[top_word_ids]))
        wc.to_file(output_d + str(i) + '.png')
