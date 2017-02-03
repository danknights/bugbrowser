import numpy as np
from util.url_helpers import url_text, parse_xml
import time
from bs4 import BeautifulSoup

PUBMED_SEARCH_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax={}&retstart={}'
PUBMED_ID_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={}'


def getIdList(term, count=-1, retstart=0, retmax=100000):
    bug_page_text = url_text(PUBMED_SEARCH_URL, term, retmax, retstart)
    if count == -1:
        count = getCount(bug_page_text)
    print('Now processing ids {} - {} for {}'.format(retstart, min(count, retstart + retmax), term))
    ids = parse_xml(bug_page_text, 'id')
    # sleep to comply with ncbi API usage rules
    time.sleep(0.3)
    if retstart + retmax > count:
        return ids
    return ids+getIdList(term, count, retstart + retmax, retmax)

def getCount(page_text):
    if page_text == "":
        return 0
    try:
        count_result = parse_xml(page_text, 'count')
        if len(count_result) > 0:
            return int(parse_xml(page_text, 'count')[0])
        else:
            return 0
    except ValueError:
        return 0

def getIdCount(bug):
    # Retrieves the first occurance of the count tag of ncbi search result, corresponds to total number of results
    try:
        count = parse_xml(url_text(PUBMED_SEARCH_URL, bug, 0, 0), 'count')[0]
        time.sleep(0.3)
        return int(count)
    except ValueError:
        print("Failed to retrieve count for bug: " + bug)
        return 0


def getIds(term):
    # wrapper function to grab maximum 100000 results per query
    idList = getIdList(term)
    return np.array(idList, int)


def getAbstract(id):
    return ' '.join(parse_xml(url_text(PUBMED_ID_URL, id), 'abstracttext'))

def writeIDs(bug_file_name, output_file_name):
    with open(bug_file_name) as bug_file, open(output_file_name, 'w') as id_file:
        for bug in bug_file:
            id_list = getIdList(bug.strip())
            for id in id_list:
                id_file.write(str(id)+'\n')

def getAbstracts(array, start, end):
    slice = array[start:end]
    id_string = ','.join(str(id) for id in slice)
    url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={}'
    full_text = url_text(url, id_string)
    soup = BeautifulSoup(full_text)
    articles = soup.find_all('pubmedarticle')
    abstracts = []
    for article in articles:
        id = article.pmid.string
        abstract = ' '.join(abstract_tag.string if abstract_tag.string is not None else "" for abstract_tag in article.find_all('abstracttext'))
        if abstract != "":
            abstracts.append('{}\t{}\n'.format(id, abstract.encode('ascii', 'ignore')))
    return abstracts

def writeAllAbstracts(id_file_name, abstract_file_name):
    with open(id_file_name) as id_file, open(abstract_file_name, 'w') as abstract_file:
        unique_ids = np.unique(np.fromfile(id_file, dtype=int, sep='\n'))
        total_ids = unique_ids.size
        start = 0
        # step size to conform with NCBI rule of requests with ~200 ids max
        step = 500
        while start < unique_ids.size:
            print('{} - {} of {} - {}%'.format(start, start + step, total_ids, round(float(start)/total_ids, 4)))
            try:
                abstracts = getAbstracts(unique_ids, start, start+step)
            except Exception:
                time.sleep(10)
                abstracts = getAbstracts(unique_ids, start, start+step)
            for abstract in abstracts:
                abstract_file.write(abstract)
            start += step



import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/tibbarrellik/Drive/Documents/KnightsLab/knightslab-repo'])

#writeIDs('../doc/bug_list.txt', '../doc/ids.txt')
writeAllAbstracts('../doc/ids.txt', '../doc/abstracts.txt')