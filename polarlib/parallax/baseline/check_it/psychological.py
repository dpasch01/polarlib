# -*- coding: utf-8 -*-

from __future__ import division
import nltk, string, math, csv, parse, os, random, re, numpy, sys, yaml
from pprint import pprint
from nltk.stem.porter import PorterStemmer
from itertools import groupby
from gensim.parsing.preprocessing import stem_text

pos = set() 
neg =  set()
lit = set()
un = set() 
weak = set()
strong = set()

slang = set()
bad = set()
common = set()
emotone = set()
emotions = set()
hiliu_neg = set()
hiliu_pos = set()

affin = {}

posfile = open('dictionaries/positive.csv', 'r')
negfile = open('dictionaries/negative.csv', 'r')
litfile = open('dictionaries/litigious.csv', 'r')
unfile = open('dictionaries/uncertainty.csv', 'r')
weakfile = open('dictionaries/modalWeak.csv', 'r')
strongfile = open('dictionaries/modalStrong.csv', 'r')

slangfile = open('dictionaries/slang.csv', 'r')
badfile = open('dictionaries/badwords.csv', 'r')
commonfile = open('dictionaries/commonwords.csv', 'r')
emotonefile = open('dictionaries/emotionaltonewords.csv', 'r')
emotionsfile = open('dictionaries/emotionwords.csv', 'r')
hiliu_negfile = open('dictionaries/hu_liu_negative.csv', 'r')
hiliu_posfile = open('dictionaries/hu_liu_positive.csv', 'r')
afinnfile = open('dictionaries/afinn.csv', 'r')

reader1 = csv.reader(posfile)
reader2 = csv.reader(negfile)
reader3 = csv.reader(litfile)
reader4 = csv.reader(unfile)
reader5 = csv.reader(weakfile)
reader6 = csv.reader(strongfile)

reader7 = csv.reader(slangfile)
reader8 = csv.reader(badfile)
reader9 = csv.reader(commonfile)
reader10 = csv.reader(emotonefile)
reader11 = csv.reader(emotionsfile)
reader12 = csv.reader(hiliu_negfile)
reader13 = csv.reader(hiliu_posfile)

def remove_punct(s):
	s = ''.join(ch for ch in s if ch not in string.punctuation)
	return s

def process(s):
	s = s.strip()
	s = s.lower()
	s = remove_punct(s)
	s = stem_text(s)

	return s

for row in reader1:
	if len(row) == 1:
		pos.add(process(row[0]))
for row in reader2:
	if len(row) == 1:
		neg.add(process(row[0]))
for row in reader3:
	if len(row) == 1:
		lit.add(process(row[0]))
for row in reader4:
	if len(row) == 1:
		un.add(process(row[0]))
for row in reader5:
	if len(row) == 1:
		weak.add(process(row[0]))
for row in reader6:
	if len(row) == 1:
		strong.add(process(row[0]))

for row in reader7:
	if len(row) == 1:
		slang.add(process(row[0]))
for row in reader8:
	if len(row) == 1:
		bad.add(process(row[0]))
for row in reader9:
	if len(row) == 1:
		common.add(process(row[0]))
for row in reader10:
	if len(row) == 1:
		emotone.add(process(row[0]))
for row in reader11:
	if len(row) == 1:
		emotions.add(process(row[0]))
for row in reader12:
	if len(row) == 1:
		hiliu_neg.add(process(row[0]))
for row in reader13:
	if len(row) == 1:
		hiliu_pos.add(process(row[0]))

for line in afinnfile:
	l = line.split('\t')
	affin[process(l[0])] = float(l[1])

mcdonald = {}
lg = {}
rid = {}

with open("dictionaries/loughran_mcdonald.yml", 'r') as mcdonald_yml:
	try:
		yamlfile = yaml.load(mcdonald_yml)
		for i in yamlfile:
			for key in i:
				mcdonald[key.lower()] = [process(w) for w in i.get(key)]
	except yaml.YAMLError as exc:
		print(exc)

with open("dictionaries/laver-garry.yml", 'r') as lg_yml:
	try:
		yamlfile= yaml.load(lg_yml)
		for i in yamlfile:
			for key in i:
				lg[key.lower()] = []
				for words in i.get(key):
					if isinstance(words, str):
						lg[key.lower()].append(process(words))
					else:
						for key1 in words:
							lg[key.lower() + "_" + key1.lower()] = [process(w) for w in words.get(key1)]
	except yaml.YAMLError as exc:
		print(exc)

with open("dictionaries/rid.yml", 'r') as rid_yml:
	try:
		yamlfile= yaml.load(rid_yml)
		for i in yamlfile:
			for key in i:
				rid[key.lower()] = []
				for words in i.get(key):
					if isinstance(words,str):
						rid[key.lower()].append([process(w) for w in words])
					else:
						for key1 in words:
							for words1 in words.get(key1):
								if isinstance(words1, str):
									rid[key.lower() + "_" + key1.lower()] = [process(w) for w in words.get(key1)]
								else:
									for key2 in words1:
										rid[key.lower() + "_" + key1.lower() + "_" + key2.lower()] = [process(w) for w in words1.get(key2)]

	except yaml.YAMLError as exc:
		print(exc)
	
def count_yml(text, yml_dic):
	yml_counts = {}

	for dic in yml_dic.keys():
		_cnt = 0
		for l in yml_dic[dic]:
			if text.find(" " + l + " ") != -1:
				_cnt = _cnt + 1
		
		yml_counts[dic] = _cnt

	return yml_counts

def _count(text, dictionary):
	_c = 0
	founds = []

	for l in list(dictionary):
		if text.find(" " + l + " ") != -1:
			founds.append(l)
			_c = _c + 1

	return _c

def count_afinn(text):
	_s = 0
	_cnt = 0
	for l in affin.keys():
		if text.find(" " + l + " ") != -1:
			_s = _s + affin[l]
			_cnt = _cnt + 1
	
	if _cnt == 0:
		_s = 0
	else:
		_s = float(_s) / float(_cnt)

	return [_cnt, _s]

def psychological_features(text):
    
	if not isinstance(text, str): text = ""
	text = process(text)
	
	features = {}

	yml_mcdonald = count_yml(text, mcdonald)
	yml_lg = count_yml(text, lg)
	yml_rid = count_yml(text, rid)

	for fk in yml_mcdonald.keys():
		features["mcdonald_" + fk] = yml_mcdonald[fk]
	for fk in yml_lg.keys():
		features["lg_" + fk] = yml_lg[fk]
	for fk in yml_rid.keys():
		features["rid_" + fk] = yml_rid[fk]

	features["negemo"] = _count(text, neg)
	features["posemo"] = _count(text, pos) 
	features["uncertain"] = _count(text, un) 
	features["litig"] = _count(text, lit) 
	features["weakModal"] = _count(text, weak) 
	features["strongModal"] = _count(text, strong) 

	# features["slang"] = _count(text, slang)
	features["bad"] = _count(text, bad) 
	features["common"] = _count(text, common) 
	features["emotone"] = _count(text, emotone) 
	features["emotions"] = _count(text, emotions) 
	features["hiliu_pos"] = _count(text, hiliu_pos) 
	features["hiliu_neg"] = _count(text, hiliu_neg) 

	affin_score = count_afinn(text)
	features['affin_count'] = affin_score[0]
	features['affin_score'] = affin_score[1]

	features = {'psy_' + k: v for k, v in features.items()}

	return features 

def main():
	textcontent = sys.argv[1]
	pprint(lexical(textcontent))

if __name__== "__main__":
	main()
