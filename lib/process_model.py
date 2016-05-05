import os, logging, json
import numpy as np
from sklearn.preprocessing import normalize
from gensim.corpora import mmcorpus
from gensim.matutils import corpus2csc
from gensim.models.ldamodel import LdaModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# model_name = sys.argv[1]  # ie 'doc/lda_model_200'
# replacementsf = sys.argv[2]
# outfname = sys.argv[3]  # ie 'model_name_output.json'
# topicn = int(sys.argv[4])  # 10
# wordn = int(sys.argv[5])  # 20
# docn = int(sys.argv[6])  # 5
def create_json(model_name, model_location, replacementsf, docsXtopicsf, docsf, bugsf, corpus_name, wordn, topicn, docn):
    output_dir = '../browser/json/'+model_name+'/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = LdaModel.load(model_location)
    bug_to_id = json.loads(open(replacementsf).read())
    # invert to get id<->bug map
    id_to_bug = {v: k for k, v in bug_to_id.items() if "." not in k}
    # load the documents X words and documents X topics matrices
    corpus = mmcorpus.MmCorpus(corpus_name)
    corpus_sparse = corpus2csc(corpus, num_terms=len(model.id2word.token2id)).T
    docsXtopics = mmcorpus.MmCorpus(docsXtopicsf)
    docsXtopics_sparse = corpus2csc(docsXtopics).T
    doc_ids = np.loadtxt(docsf, dtype=np.int64, delimiter='\t', usecols=(0,))
    titles = np.loadtxt(docsf, dtype=str, delimiter='\t', usecols=(1,))
    bugs = []
    for bug in open(bugsf):
        bug = bug.strip()
        print bug
        if not "." in bug and " " in bug and bug in bug_to_id:
            print 'made it'
            # bug is genus and species and has assigned id
            bug_id = bug_to_id[bug]
            if bug_id in model.id2word.token2id:
                model_id = model.id2word.token2id[bug_id]
                bugs.append((bug, model_id))
    print bugs
    bugExpELogBeta = (model.expElogbeta)[:, zip(*bugs)[1]]
    expElogbeta_row = normalize(bugExpELogBeta, axis=0)
    expElogbeta_col = normalize(bugExpELogBeta, axis=1)
    bug_dict = {}
    for i, (bug, model_id) in enumerate(bugs):
        print bug
        # if 'faecalibacterium' not in bug:
        #	continue
        bug_row = expElogbeta_row[:, i]
        bug_col = expElogbeta_col[:, i]
        topic_array = np.sqrt(bug_row * bug_col)
        top_topics = [int(e[0]) for e in sorted(enumerate(topic_array), key=lambda x: x[1], reverse=True)]
        bug_docs = corpus_sparse[:, model_id].nonzero()[0]
        bug_docsXtopics = docsXtopics_sparse[bug_docs]
        topics_dict = []
        for k, t in enumerate(top_topics[:topicn]):
            topic_dict = {}
            docs_for_topic = bug_docsXtopics[:, t].A.T[0]
            sorted_docs_indices = docs_for_topic.argsort()[::-1]
            doc_indices = bug_docs[sorted_docs_indices]
            top_docs_for_topic = doc_ids[doc_indices]
            top_docs_for_topic = top_docs_for_topic[:min(docn, len(top_docs_for_topic))]
            top_titles = titles[doc_indices]
            top_titles = top_titles[:min(docn, len(top_titles))]
            topic_dict['topdocs'] = zip(top_docs_for_topic, top_titles)
            topic_words = list(zip(*model.show_topic(t, wordn))[0])
            topic_word_ids = np.array([model.id2word.token2id[word] for word in topic_words])
            for i, word in enumerate(topic_words):
                if word in id_to_bug:
                    topic_words[i] = id_to_bug[word]
            word_docs = corpus_sparse[:, topic_word_ids]
            bug_docs_full = corpus_sparse[:, model_id]
            combined_docs = bug_docs_full.A * word_docs.A
            for i, combined_doc in enumerate(combined_docs.T):
                comb_docs_nonzero = combined_doc.nonzero()[0]
                comb_docsXtopics = docsXtopics_sparse[comb_docs_nonzero]
                comb_docs_for_topic = comb_docsXtopics[:, t].A.T[0]
                comb_sorted_docs_indices = comb_docs_for_topic.argsort()[::-1]
                comb_doc_indices = comb_docs_nonzero[comb_sorted_docs_indices]
                comp_top_docs_for_topic = doc_ids[comb_doc_indices]
                comp_top_docs_for_topic = comp_top_docs_for_topic[:min(docn, len(comp_top_docs_for_topic))]
                comp_titles = titles[comb_doc_indices]
                comp_titles = comp_titles[:min(docn, len(comp_titles))]
                if len(comp_top_docs_for_topic) > 0:
                    topic_dict[topic_words[i]] = zip(comp_top_docs_for_topic, comp_titles)
            topics_dict.append((t, topic_dict))
        bug_dict[bug] = topics_dict
        with open(output_dir+bug+'.json', 'w') as outf:
            outf.write(json.dumps(bug_dict[bug], indent=4, separators=(',', ': ')))
    # with open(outfname, 'w') as outf:
    #     outf.write(json.dumps(bug_dict, indent=4, separators=(',', ': ')))
    # with open('bugs.json', 'w') as outf:
    #     outf.write(json.dumps(zip(*bugs)[0], indent=4, separators=(',', ': ')))

