import os, logging, json, glob
import numpy as np
from sklearn.preprocessing import normalize
from gensim.corpora import mmcorpus
from gensim.matutils import corpus2csc
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def create_json(files, wordn, topicn, docn):
    docsXtopics, docsXwords, id_to_bug, model, output_dir = setup(files)

    # store document ids and document titles in numpy arrays, note these must be kept in sync
    doc_ids = np.loadtxt(files.abstracts, dtype=int, delimiter='\t', usecols=(0,))
    titles = np.loadtxt(files.abstracts, dtype=str, delimiter='\t', usecols=(1,))

    # get the model id of every bug retained in the model
    bug_to_model_id = [(bug, model.id2word.token2id[id]) for id, bug in id_to_bug.items() if id in model.id2word.token2id]

    # get the normalized topicsXbugs arrays
    topicsXbugs_colnorm, topicsXbugs_rownorm = topicsXbugs_norms(bug_to_model_id, model)

    bug_dict = {}
    for col, (bug, model_id) in tqdm(enumerate(bug_to_model_id)):
        # perform RAR computation
        bug_row = topicsXbugs_rownorm[:, col]
        bug_col = topicsXbugs_colnorm[:, col]
        topic_array = np.sqrt(bug_row * bug_col)

        # extract top topics by id from RAR
        top_topics = topic_array.argsort()[::-1]

        topics_dict = []
        docs_for_bug = docsXwords[:, model_id]
        for topic in top_topics[:topicn]:
            topic_dict = {}
            # create entry of top documents overall for this bug and topic
            top_docs_for_topic, top_titles_for_topic = top_documents(docsXtopics, docs_for_bug, titles, doc_ids, topic, docn)
            topic_dict['topdocs'] = list(zip(top_docs_for_topic, top_titles_for_topic))

            # get the ids of the words that occur in this topic, converting bug terms as necessary
            topic_words = list(zip(*model.show_topic(topic, wordn)))[0]
            topic_word_ids = np.array([model.id2word.token2id[word] for word in topic_words])
            topic_words = [id_to_bug[word] if word in id_to_bug else word for word in topic_words]

            # get subset of docsXwords matrix corresponding to the words of this topic
            docs_for_words = docsXwords[:, topic_word_ids]

            # find intersection of documents mentioning this bug, and documents mentioning each word in the topic
            docs_for_bug_and_words = docs_for_bug.A * docs_for_words.A
            for bug_col, docs_for_bug_and_word in enumerate(docs_for_bug_and_words.T):
                top_docs_for_topic_bug_and_word, top_titles_for_topic_bug_and_word = top_documents(docsXtopics, docs_for_bug_and_word, titles, doc_ids, topic, docn)
                # create entry of top documents for this bug, this topic, and this specific word in the topic
                # only store if there was at least 1 document using both this bug and this word
                if len(top_docs_for_topic_bug_and_word) > 0:
                    topic_dict[topic_words[bug_col]] = list(zip(top_docs_for_topic_bug_and_word, top_titles_for_topic_bug_and_word))
            topics_dict.append((int(topic), topic_dict))
        bug_dict[bug] = topics_dict
        with open(output_dir+bug+'.json', 'w') as outf:
            outf.write(json.dumps(bug_dict[bug], indent=4, separators=(',', ': ')))
    b = [bug.split(".")[0] for bug in glob.glob("*.json")]
    with open('bugs.json', 'w') as outf:
        outf.write(json.dumps(b))


def topicsXbugs_norms(bug_to_model_id, model):
    # extract subarray of the topicsXbugs from the general topicsXwords <-> model.expElogbeta
    topicsXbugs = (model.expElogbeta)[:, list(zip(*bug_to_model_id))[1]]  # retain only columns corresponding to bugs
    #  normalize by rows and columns for RAR computation
    topicsXbugs_rownorm = normalize(topicsXbugs, axis=0)
    topicsXbugs_colnorm = normalize(topicsXbugs, axis=1)
    return topicsXbugs_colnorm, topicsXbugs_rownorm


def setup(files):
    # setup the output directory
    base_model_name = os.path.splitext(os.path.basename(files.model))[0]
    output_dir = '../browser/json/' + base_model_name + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load the topic model
    model = LdaModel.load(files.model)
    # load replacements used
    bug_to_id = json.loads(open(files.replacements).read())
    # invert to id<->bug map, ditching s. genus terms
    id_to_bug = {v: k for k, v in bug_to_id.items() if "." not in k}
    # load the docsXwords and docsXtopics matrices (in sparse format)
    corpus = mmcorpus.MmCorpus(files.corpus)
    docsXwords_sparse = corpus2csc(corpus, num_terms=len(model.id2word.token2id)).T
    docsXtopics = mmcorpus.MmCorpus(files.docsXtopics)
    docsXtopics_sparse = corpus2csc(docsXtopics).T
    return docsXtopics_sparse, docsXwords_sparse, id_to_bug, model, output_dir


def top_documents(docsXtopics, docs_for_term, titles, doc_pmed_ids, topic, max_docs):
    # find ids of documents that mention the term
    term_doc_idx = docs_for_term.nonzero()[0]
    # for these documents, find which have the highest proportion of the current topic
    term_docsXtopics = docsXtopics[term_doc_idx]
    topic_weights_for_docs = term_docsXtopics[:, topic].A.T[0]
    top_docs_for_topic_idx = topic_weights_for_docs.argsort()[::-1]
    top_word_doc_idx = term_doc_idx[top_docs_for_topic_idx]
    # get up to docn of the pmed ids of these documents, and the corresponding titles
    doc_count = len(top_word_doc_idx)
    top_documents = doc_pmed_ids[top_word_doc_idx][:min(max_docs, doc_count)]
    top_documents = [int(x) for x in top_documents] # make sure this can be converted by json
    top_titles = titles[top_word_doc_idx][:min(max_docs, doc_count)]
    return top_documents, top_titles
