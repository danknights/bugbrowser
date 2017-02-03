from run_lda import run_lda, get_tfidf, get_parameters


def main():
    # configure logging for gensim
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # parse command line parameters
    args = getArgs()
    # extract file paths and parameters from command line arguments
    corpusf, dictf, docsf, replacementf, model_dir, bugf, modelf, docsXtopicsf, base_model_name = getFilePaths(args)
    docn, eta, alpha, num_topics, topicn, wordn = get_parameters(args)
    # set up corpus and dictionary, create if necessary
    tfidf_model = get_tfidf(bugf, corpusf, dictf, docsf, replacementf, args['update'])
    # run LDA on the corpus, save model
    run_lda(alpha, tfidf_model, docsXtopicsf, eta, modelf, num_topics, word_dict)
    # create word cloud for each topic
    #make_clouds(modelf, base_model_name, replacementf, wordn)
    # create json files for each bug for use in browser
    #create_json(base_model_name, modelf, replacementf, docsXtopicsf, docsf, bugf, corpusf, wordn, topicn, docn)


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

#python main.py -u -b ../doc/bug_list.txt -f ../doc/abstracts/faecalibacterium\ prausnitzii.pmed -d practice.dict -c practice.corpus -m practice -t 10 -r ../doc/replacements.json
