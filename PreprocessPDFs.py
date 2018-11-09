'''
PES Question-Answer Bot

Preprocessing Program.

Tanya Mehrotra 01FB15ECS323 
God. 01FB15ECS339 

Make sure to compile the program and then run.

Compile by: <path to pypy3> -O -m py_compile PreprecessPDFs.py 

'''

# from __future__ import print_function, division

# ElementTree for building the tree of the XML File
import xml.etree.ElementTree as ElementTree

# To get arguments about which folder the textbooks are in
import sys

# Stripping off HTML tags from text fields
import re

# Removing punctuation which cleaning strings
import string

# Used in cosine similarity
import math

# Standard array calculations
import numpy

# Store the preprocessing
import pickle

# To run shell commands
import os

# Used in cosine similarity, word count in sentences
from collections import Counter

# Parallelizing the tasks
from multiprocessing.dummy import Pool as ThreadPool

# Pretty print a progress bar
from tqdm import tqdm

# To store the distance matrix as a sparse matrix
from scipy.sparse import bsr_matrix 

# NTLK Modules
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import ne_chunk
from nltk.tree import Tree

# To read a directory
from glob import glob


# Takes the text, and removes any tags if present
def StripHTMLTags(raw_html):
	cleaner = re.compile('<.*?>')
	cleantext = re.sub(cleaner, '', raw_html)

	return cleantext

# Cleans the string by removing punctuations and stopwords. Also stems each
# word to get it to its basic form
stopWords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
porterStemmer = PorterStemmer()
def CleanSentence(sentence, removeStopWords = True, addSynonyms = False):
	cleaned_sentence = []

	# preprocessing, could swap porter stemmer for wordnet lemmatizer
	words = word_tokenize(sentence)
	tagged_words = pos_tag(words)

	for word, pos in tagged_words:
		if word in string.punctuation:
			continue

		if removeStopWords and (word in stopWords):
			continue
		
		cleaned_sentence.append(porterStemmer.stem(word).lower())

		if addSynonyms:
			synonyms = [syn.name().split('.')[0] for syn in wordnet.synsets(word.lower())]
			cleaned_sentence += synonyms

	if addSynonyms:
		return list(set(cleaned_sentence))
	else:
		return cleaned_sentence

# Finds the similarity between 2 sentences based on cosine similarity and the
# distance between the sentences
clusters = []
similarity_threshold = 0.5
def SentenceSimilarity(t1, t2):
	vec1 = cleaned_sentences_with_counter[t1]
	vec2 = cleaned_sentences_with_counter[t2]

	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum(map(lambda x: vec1[x] * vec2[x], intersection))

	sum1 = sum(map(lambda x: vec1[x]**2, vec1.keys()))
	sum2 = sum(map(lambda x: vec2[x]**2, vec2.keys()))
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	cosine = 0.0 if (not denominator) else float(numerator) / denominator

	sentenceDistance = abs(int(t1) - int(t2))
	difference = 1 - (sentenceDistance / 10) if sentenceDistance < 10 else 0
	
	similarity = 0.5*cosine + 0.5*difference

	if (similarity > similarity_threshold):
		added = False
		for cluster in clusters:
			if t1 in cluster:
				added = True
				cluster.add(t2)
			elif t2 in cluster:
				added = True
				cluster.add(t1)
		else:
			if not added:
				clusters.append(set([t1, t2]))

	similarityMatrix[t1][t2] = similarity
	similarityMatrix[t2][t1] = similarity

	try:
		tqdm_iterator.__next__()
	except:
		pass

if (len(sys.argv) == 1):
	print("Usage: <directory for pypy> ./PreprocessPDFs.py <list of directories>")
	exit(0)
else:
	directoryNames = sys.argv[1:]

pdfString = ""

print("Collecting all PDFs together")
print("----------------------------")

pdfs = []

for directoryName in directoryNames:
	if (directoryName[-1] == "/"):
		directoryName = directoryName[:-1]

	pdfs += glob(directoryName + "/*.pdf")

print("Done\n\n")

print("Converting PDFs to XMLs")
print("-----------------------")

pool = ThreadPool()

pool.map(lambda x: os.system("pdftohtml -xml \"" + x + "\" \"" + x + ".xml\" > /dev/null"), tqdm(pdfs))

try:
	pool.join()
except:
	pass

pool.close()

del pool

print("Done\n\n")

print("Extracting string from all XMLs")
print("-------------------------------")

for pdf in tqdm(pdfs):
	tree = ElementTree.ElementTree(file = pdf + ".xml")
	root = tree.getroot()

	for element in root.getiterator():
		if (element.tag == 'text'):
			try:
				pdfString += (" " + StripHTMLTags(element.text))
			except:
				pass

print("Done\n\n")

print("Clustering Sentences")
print("--------------------")

sentences = sent_tokenize(pdfString)

cleaned_sentences = [CleanSentence(sentence) for sentence in sentences]
cleaned_sentences_with_counter = [Counter(sentence) for sentence in cleaned_sentences]

similarityMatrix = numpy.zeros(shape = (len(sentences), len(sentences)))

# Global iterator for the progress bar
tqdm_iterator = tqdm(range(len(sentences) * (len(sentences) - 1) // 2)).__iter__()

pool1 = ThreadPool()
pool2 = ThreadPool()

pool1.map(lambda x: pool2.map(lambda y: SentenceSimilarity(x,y), range(x+1, len(sentences))), range(len(sentences)-1))

pool1.close()
pool1.join()

pool2.close()
pool2.join()

for _ in tqdm_iterator:
	pass

similarityMatrixSparse = bsr_matrix(similarityMatrix)

clusters = [tuple(sorted(list(cluster))) for cluster in clusters]
clusters = sorted(list(set(clusters)))

# print(clusters)
# print(len(clusters))

print("Done\n\n")

print("Named Entity Propagation")
print("------------------------")

def entityForEachSentence(sent):
	chunked = ne_chunk(pos_tag(word_tokenize(sent)))
	prev = None
	continous_chunk = []
	current_chunk = []
	for i in chunked:
		if type(i) == Tree:
			current_chunk.append(" ".join([token for token, pos in i.leaves()]))
		elif current_chunk:
			named_entity = " ".join(current_chunk)
			if named_entity not in continous_chunk:
				continous_chunk.append(named_entity)
				current_chunk = []
		else:
			continue
	return continous_chunk

entityAllSentences = list(map(lambda x: entityForEachSentence(x.lower()), tqdm(sentences)))
for i in range(1,len(sentences)):
	entityAllSentences[i] += entityAllSentences[i-1]

print("Done\n\n")

print("Extracting important words")
print("--------------------------")

words = sorted(list(set([word for sentence in cleaned_sentences for word in sentence] + [word for sentence in entityAllSentences for word in sentence])))

print("Done\n\n")


print("Creating Search Matrix")
print("----------------------")

searchMatrix = numpy.zeros(shape = (len(clusters), len(words)))

for i, cluster in enumerate(clusters):
	clusterWords = set([word for index in cluster for word in cleaned_sentences[index] + entityAllSentences[index]])
	for word in clusterWords:
		searchMatrix[i][words.index(word)] = 1

searchMatrixSparse = bsr_matrix(searchMatrix)

print("Done\n\n")

print("Saving Required Files")
print("---------------------")

fileObj = open('Clusters', 'wb')
pickle.dump(clusters, fileObj)
fileObj.close()

fileObj = open('Sentences', 'wb')
pickle.dump(sentences, fileObj)
fileObj.close()

fileObj = open('Cleaned_Sentences', 'wb')
pickle.dump(cleaned_sentences, fileObj)
fileObj.close()

'''
fileObj = open('Similarity_Matrix', 'wb')
pickle.dump(similarityMatrix, fileObj)
fileObj.close()
'''

fileObj = open('Similarity_Matrix_Sparse', 'wb')
pickle.dump(similarityMatrixSparse, fileObj)
fileObj.close()

fileObj = open('Important_Words', 'wb')
pickle.dump(words, fileObj)
fileObj.close()

'''
fileObj = open('Search_Matrix', 'wb')
pickle.dump(searchMatrix, fileObj)
fileObj.close()
'''

fileObj = open('Search_Matrix_Sparse', 'wb')
pickle.dump(searchMatrixSparse, fileObj)
fileObj.close()

weights = numpy.ones(shape=(len(clusters),))

fileObj = open('Weights', 'wb')
pickle.dump(weights, fileObj)
fileObj.close()

print("Done\n\n")