import glob, os
import re
import numpy as np
from nltk.corpus import stopwords
from numpy import linalg as LA

# Returns the alphanumerical words of a text as a list.
def words(text): return re.findall(r'\w+', text)

pairs = {}
terms = {}

# Reads the files in a directory.
def reader(directory):
	os.chdir(directory)
	stop = set(stopwords.words('english'))
	termIdx = 0;
	for file in glob.glob("*.txt"):
		corpus = words(open(file).read())
		for idx, w in enumerate(corpus):
			if w not in stop:
				i = idx+1;
				if w not in pairs:
					pairs[w] = {}
				if w not in terms:
					terms[w] = termIdx
					termIdx += 1
				while i<idx+5 and i<len(corpus):
					w2 = corpus[i]
					#print("i: ", w2)
					if w2 not in stop:
						if w2 not in pairs:
							pairs[w2] = {}
						if w2 not in terms:
							terms[w2] = termIdx
							termIdx += 1
						if w2 not in pairs[w]: pairs[w][w2] = 1
						else: pairs[w][w2] += 1
						if w not in pairs[w2]: pairs[w2][w] = 1
						else: pairs[w2][w] += 1

					i = i + 1	

# Constructs the graph of words.
def construct():
	n = len(terms)
	global wordMatrix
	wordMatrix = np.zeros((n,n))
	for key1, val in pairs.items():
		for key2, val2 in val.items():
			id1 = terms[key1]
			id2 = terms[key2]
			wordMatrix[id1][id2] = val2

# Power iteration algorithm for finding the eigenvalues.
def powerIteration(A, n):
	v = np.random.rand(len(A),1)
	print(len(A))
	v = v/LA.norm(v)
	for i in range(0,n):
		w = np.dot(A, v)
		v = w/LA.norm(w)
	return v

# Gets the n maximum elements of numpy vector. Returns the values. 
def getNMax(v, n):
	v = np.transpose(v)
	a = np.squeeze(np.asarray(v))
	ind = np.argpartition(a, -n)[-n:]
	ind = ind[np.argsort(a[ind])]
	#print(ind)
	ind = ind[::-1]
	return ind

# Gets the words indexed by ind from the dictionary.
def getMaxWords(ind):
	l = list(terms.keys())
	res = []
	for _,i in enumerate(ind):
		r = l[list(terms.values()).index(i)]
		res.append(r)
	print(res)

def main():
	cwd = os.getcwd()
	reader("data/train/pos/") #positives
	construct()
	v = powerIteration(wordMatrix, 30)
	maxes = getNMax(v, 100)
	getMaxWords(maxes)
	global pairs
	global terms
	pairs = {}
	terms = {}
	os.chdir(cwd)
	reader("data/train/neg/") #negatives
	construct()
	v = powerIteration(wordMatrix, 30)
	maxes = getNMax(v, 100)
	getMaxWords(maxes)

main()