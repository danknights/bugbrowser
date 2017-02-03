import os, random, string
from itertools import product
from random import sample
from collections import Counter
from json import dumps, loads
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def create_bug_frequency(bug_file):
    bugs = [line.strip() for line in open(bug_file, 'r')]
    unique_genus_species = [bug for bug in bugs if len(bug.split()) == 2]
    bug_frequencies = Counter()
    for line in tqdm(open('abstract_samples.txt').readlines()):
        for bug in unique_genus_species:
            genus, species = bug.split()
            line = line.lower()
            # check if either 'genus species' or 'g. species' occurs in document
            if bug in line or "{}. {}".format(genus[0], species) in line:
                bug_frequencies[bug] += 1
    return bug_frequencies


def get_bug_frequency(bug_frequency_json_filename):
    with open(bug_frequency_json_filename) as f:
        return Counter(loads(f.read()))


def bugs_exceeding_count(bug_filename, bug_frequency_filename, min_count):
    if not os.path.isfile(bug_frequency_filename):
        freq = create_bug_frequency(bug_filename)
        with open(bug_frequency_filename, 'w') as f:
            f.write(dumps(freq))
    else:
        freq = get_bug_frequency(bug_frequency_filename)
    return [bug for bug in freq if freq[bug] >= min_count]


# bug_over_ten = bugs_exceeding_count('bug_list.txt', '../lib/bug_frequency.json', 10)
# print(len(bug_over_ten))

def generate_pbs_scripts():
    base_pbs_contents = """#!/bin/bash -l
#PBS -l walltime=12:00:00,nodes=1:ppn=32,mem=500gb
#PBS -m abe
#PBS -M meule012@umn.edu
module load python-epd
source ~/mypython/bin/activate
cd ~/knightslab-repo/lib
python run_lda.py -u -b ../doc/bug_list.txt -f ../doc/abstracts_with_titles.txt -d titled.dict -c titled.corpus -m titled_{t}_{a}_{b} -t {t} -r ../doc/replacements.json -a {a} -e {b} &> titled_{t}_{a}_{b}_output.txt"""
    for (t, a, b) in product(range(90, 120, 10), [50, 100, 'symmetric', 'auto'], [.01, .1, .25, None]):
        with open('titled_{t}_{a}_{b}.pbs'.format(t=t, a=a, b=b), 'w') as pbsf:
            pbsf.write(base_pbs_contents.format(t=t, a=a, b=b))


def random_file_sample(inFile, outFile, percent=10):
    with open(inFile, 'r') as fin:
        with open(outFile, 'w') as fout:
            lines = fin.readlines()
            fout.writelines(sample(lines, len(lines) / int(percent)))


def make_frequency_plot(bug_frequency_json, n=30):
    sns.set_style("whitegrid")
    bugs = Counter(loads(open(bug_frequency_json).read()))
    labels, counts = zip(*bugs.most_common(n))
    sns.barplot(x=labels, y=counts).set_xticklabels(labels, rotation=90)
    plt.show()

def replace_all(doc, replacements):
    for i, j in replacements.items():
        doc = doc.replace(i, j)
    return doc


def get_replacements(bug_list):
    ids = set()
    replacements = {}
    for bug in bug_list:
        while True:
            uid = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
            if not uid in ids:
                ids.add(uid)
                break
        genus, species = bug.split()
        replacements[genus + " " + species] = uid
        replacements[genus[0] + ". " + species] = uid
    return replacements

#print(get_replacements(bugs_exceeding_count('../doc/bug_list.txt', '../doc/bug_frequency.json', 10)))
