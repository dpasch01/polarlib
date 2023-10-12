from __future__ import division

from nltk.stem.porter import PorterStemmer
from itertools import groupby
import sys, nltk, string, math, re

def vocabulary_richness(text):

	if not isinstance(text, str): text = ""
	text = nltk.word_tokenize(text)
	text = [w.lower().rstrip(string.punctuation).lstrip(string.punctuation) for w in text]
	vocabulary_richness = {}

	def hapax(s):
		legomena = filter(lambda x: s.count(x) == 1, s)
		legomena = list(filter(None, legomena))

		dislegomena = filter(lambda x: s.count(x) == 2, s)
		dislegomena = list(filter(None, dislegomena))
		
		vocabulary_richness["vr_happax_legomena"] = len(legomena)
		vocabulary_richness["vr_happax_dislegomena"] = len(dislegomena)

	def TTR(s):
		stemmer = PorterStemmer()
		N = len(s)
		distinct = set([stemmer.stem(token) for token in s])
		V = len(distinct)
		try:
			TTR = V/N
		except ZeroDivisionError:
			TTR = 0            
		vocabulary_richness["vr_ttr"] = TTR

	def yuleK(s):
		d = {}
		stemmer = PorterStemmer()
		for w in s:
			w = stemmer.stem(w).lower()
			try:
    				d[w] += 1
			except KeyError:
    				d[w] = 1

		M1 = float(len(d))
		M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))])

		try:
			vocabulary_richness["vr_yules_k"] = (M1*M1)/(M2-M1)
		except ZeroDivisionError:
			vocabulary_richness["vr_yules_k"] = 0

	def sichelS(s):
		n = len(s)
		s2 = sorted(s)
		v2 = V(2, s2, n)
			
		if n > 0: 
    			S = float(v2)/float(n)
		else:
    			S = 0
		
		vocabulary_richness["vr_sichel_s"] = S             

	def V(num, s1, n):
		vm = 0
		for i in range(0, n-1):
			if i == 0:
				flag = False
				for j in range(0, num):
					if s1[i] == s1[i + j]:
						flag = True 
					else:
						flag = False 
				if flag == True:
					vm += 1
			elif s1[i] != s1[i-1]:
				flag = False
				for j in range(0, num):
					if i + j < n:
						if s1[i] == s1[i + j]:
							flag = True 
						else:
							flag = False 
				if flag == True:
					vm += 1
		return vm
    
	def brunetsW(s):
		stemmer = PorterStemmer()
		a = -0.169
		N = len(s)

		distinct = set([stemmer.stem(token) for token in s])
		V = len(distinct)
		if V == 0:
			W=0
		else:
			W = N**(V**a)

		vocabulary_richness["vr_brunets_w"] = W

	def simpsonD(s):
		s1 = sorted(s)
		n = len(s)
		D = 0
		
		for m in range(1, n):
    			D += V(m, s1, n) * (m / n) * ((m - 1)/(n-1))

		vocabulary_richness["vr_simpson_d"] = D

	def honoreR(s):
		n = len(s)
		s2 = sorted(s)
		v1 = V(1, s2, n) 
		if n > 0:
    			R = 100 * (math.log(n)/ (1 - v1/n))
		else:
    			R = 0

		vocabulary_richness["vr_honore_r"] = R

	hapax(text)
	TTR(text)
	brunetsW(text)
	honoreR(text)
	sichelS(text)
	# simpsonD(text)
	yuleK(text)

	return vocabulary_richness 

def main():
	textcontent = sys.argv[1]
	print(vocabulary_richness(textcontent))
  
if __name__== "__main__":
	main()
