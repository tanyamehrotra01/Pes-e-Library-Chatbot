import json
import nltk
from nltk.tokenize import sent_tokenize
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
import os
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import math
from textblob import TextBlob as tb
import numpy
import sys
import pickle
import scipy

listOfQuestions = []

inputFileName = sys.argv[1]
with open(inputFileName,"r") as f:
	data = f.read()
	questions = data.split("\n\n")

	for i in range(0,len(questions)):
		ques = {}
		contents = questions[i].split("\n")
		if(len(contents) > 4):
			ques['type'] = "MCQ"
		else:
			ques['type'] = "SUB"

		question = nltk.sent_tokenize(contents[0])
		if(len(question) == 2):
			ques['question'] = question[1]
		else:
			q = ""
			for j in range(1,len(question)):
				q = q + question[j] + " "
			ques['question'] = q

		if(ques['type'] == "MCQ"):
			optionsArray = []
			for o in range(1,len(contents)):
				op = contents[o][3:]
				optionsArray.append(op)
			ques['options'] = optionsArray
		else:
			ques['options'] = "NA"
		listOfQuestions.append(ques)

	# print(listOfQuestions)

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

	def AnswerQuestions(passage, questions):

		def TFIDF(sentences):

			def tf(word, blob):
				return blob.words.count(word) / len(blob.words)

			def n_containing(word, bloblist):
				return sum(1 for blob in bloblist if word in blob.words)

			def idf(word, bloblist):
				return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

			def tfidf(word, blob, bloblist):
				return tf(word, blob) * idf(word, bloblist)

			if type(sentences[0]) == type(list()):
				bloblist = [tb(" ".join(sentence)) for sentence in sentences]
			else:
				bloblist = [tb(sentence) for sentence in sentences]
			
			scores = [{word: tfidf(word, blob, bloblist) for word in blob.words} for i, blob in enumerate(bloblist)]

			return scores

		def SentenceSimilarities(sentences, tfidfScores, one_sentence, weighted = False):
			
			if (type(sentences[0]) != type(list())):
				for i in range(len(sentences)):
					sentences[i] = word_tokenize(sentences[i])

			if (type(one_sentence) != type(list())):
				one_sentence = word_tokenize(one_sentence)

			scores = []
			i = 0

			for sentence in sentences:
				current_score = 0
				for word in one_sentence:
					if word in sentence:
						if weighted == False:
							try:
								current_score += (1 * tfidfScores[i][word])
							except:
								pass
						else:
							try:
								current_score += (1 * tfidfScores[i][word]) * (1 / (1 + abs(i - weighted)))
							except:
								pass
				scores.append(current_score)
				i += 1

			return scores

		sentences_in_passage = sent_tokenize(passage)
		cleaned_sentences_in_passage = [CleanSentence(sentence) for sentence in sentences_in_passage]

		cleaned_sentences_without_removing_stopwords = [CleanSentence(sentence, removeStopWords = False) for sentence in sentences_in_passage]
		passage_sentence_scores = TFIDF(cleaned_sentences_without_removing_stopwords)

		answers = []

		for question, options in questions:
			choice_probabilities = []

			cleaned_question = CleanSentence(question)

			question_scores = SentenceSimilarities(cleaned_sentences_in_passage, passage_sentence_scores, cleaned_question)
			sentence_number = numpy.argmax(question_scores)

			for option in options:
				cleaned_option = CleanSentence(option, addSynonyms = True)
				option_scores = SentenceSimilarities(cleaned_sentences_in_passage, passage_sentence_scores, cleaned_option, weighted = sentence_number)
				choice_probabilities.append(max(option_scores))

			total_choice_probabilities = sum(choice_probabilities)
			choice_probabilities = [probability / total_choice_probabilities for probability in choice_probabilities]

			answers.append(choice_probabilities)

			#print(choice_probabilities, numpy.argmax(choice_probabilities))
			print("Option " ,numpy.argmax(choice_probabilities)+1, "is the correct option\n\n")
		return answers

	fileObj = open('Important_Words', 'rb')
	words = pickle.load(fileObj)

	fileObj = open('Search_Matrix_Sparse', 'rb')
	searchMatrixSparse = pickle.load(fileObj)

	fileObj = open('Weights', 'rb')
	weights = pickle.load(fileObj)

	fileObj = open('Clusters', 'rb')
	clusters = pickle.load(fileObj)

	fileObj = open('Sentences', 'rb')
	sentences = pickle.load(fileObj)

	for dic in listOfQuestions:
		for key,value in dic.items():
			if(key == "type"):
				questionType = dic[key]
			if(key == "question"):
				question = dic[key]
			if(key == "options"):
				options = dic[key]
		
		sentence = sent_tokenize(question)
		cleaned_sentence = [CleanSentence(_sentence) for _sentence in sentence]
		
		queryVector = numpy.zeros(shape = (len(words),))

		query_words = set([word for _sentence in cleaned_sentence for word in _sentence])
		for word in query_words:
			try:
				queryVector[words.index(word)] = 1
			except:
				pass

		clusterNumber = numpy.argmax(searchMatrixSparse * queryVector * weights)		

		clusterSentencesIndex = clusters[clusterNumber]		

		# print(clusters[clusterNumber])

		passage = " ".join([sentences[index] for index in clusters[clusterNumber]])

		print("question - ", question)
		print("options - ", options)
		#print("passage", passage)
		

		AnswerQuestions(passage,[[question,options]])