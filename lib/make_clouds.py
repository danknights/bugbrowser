import os, sys, json
import numpy as np
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
from sklearn.preprocessing import normalize
from tqdm import tqdm

model_name = 'titled_100_symmetric_None'
model_location = 'doc/titled_100_symmetric_None'
output_d = 'doc/bugbrowser/clouds/'+model_name+"/"
n_words = 50
n_topics = 5
replacementsf = 'doc/replacements.json'
d = os.getcwd()

wc = WordCloud(width=1000, height=500, background_color='white')
print 'Loading model'
model = LdaModel.load(model_location)
print 'Normalizing by topics, and by words'
expELogBeta = model.expElogbeta
pTW = normalize(expELogBeta, axis=0)
pWT = normalize(expELogBeta, axis=1)



bug_to_id = json.loads(open(replacementsf).read())
# invert to get id<->bug map
id_to_bug = {v: k for k, v in bug_to_id.items() if "." not in k}
# for bug in open('doc/practice_bugs.txt'):
# 	f_index = model.id2word.token2id[bug.strip()]
# 	w_rar = np.sqrt(pWT[:,f_index]*pTW[:,f_index])
# 	print pWT[:,f_index]
# 	print pTW[:,f_index]
# 	top_topics = w_rar.argsort()[:-1-n_topics:-1]
# 	top_topics_2 = pWT[:,f_index].argsort()[:-1-n_topics:-1]
# 	top_topics_3 = pTW[:,f_index].argsort()[:-1-n_topics:-1]
# 	print bug.strip()+': '+str(top_topics)+' - RAR'
# 	print bug.strip()+': '+str(top_topics_2)
# 	print bug.strip()+': '+str(top_topics_3)

for i in tqdm(range(len(expELogBeta))):
	t_rar = np.sqrt(pTW[i]*pWT[i])
	top_word_ids = t_rar.argsort()[:-1-n_words:-1] 
	top_words = [model.id2word.id2token[wordid] for wordid in top_word_ids]
	top_words = [id_to_bug[word] if word in id_to_bug else word for word in top_words]
	wc.fit_words(zip(top_words, t_rar[top_word_ids]))
	wc.to_file(os.path.join(d, output_d+str(i)+'.png'))


