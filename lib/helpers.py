from itertools import product
from random import sample
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
			fout.writelines(sample(lines, len(lines)/int(percent)))

def make_frequency_plot(bug_frequency_json, n=30):
	sns.set_style("whitegrid")

	bugs = Counter(json.loads(open(bug_frequency_json).read()))
	labels, counts = zip(*bugs.most_common(n))
	ax = sns.barplot(x=labels, y=counts).set_xticklabels(labels, rotation=90)
	plt.show()
