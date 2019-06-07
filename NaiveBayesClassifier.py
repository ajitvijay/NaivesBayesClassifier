import math
import sys
import json
import re
import string
import numpy as np
import time

def VocabAccumulator(sentences):
    vocab = set()
    for i in sentences:
        for word in i:
            vocab.add(word)
    return vocab

def reviewExtraction(filename, filename2): # for the test files, returns a list of list where list holds list of the reviews split up by word
    file = open(filename, 'r', encoding="utf8")
    file2 = open(filename2, 'r', encoding="utf8")
    reviews = file.read().split("<br /><br />")
    reviews2 = file2.read().split("<br /><br />")
    sentences = []
    file2_sentences = []
    for line in reviews:
        line = re.sub('[^a-zA-Z]+', ' ', line, flags=re.IGNORECASE).lower().split(" ")
        line = [w for w in line if w not in stopwords]
        sentences.append(line)
    for text in reviews2:
        text = re.sub('[^a-zA-Z]+',' ', text, flags=re.IGNORECASE).lower().split(" ")
        text = [word for word in text if word not in stopwords]
        file2_sentences.append(text)

    return sentences, file2_sentences

def getMultinomialBOW(positivereviews, negativereviews, dict_pos_total, dict_neg_total, num_pos_words, num_neg_words, num_pos_reviews, num_neg_reviews):
    positive_count = 0
    negative_count = 0
    num_positive_reviews = 0
    num_negative_reviews = 0

    #compute prior probability with number of negative and positive reviews from training set
    priorProb_pos = math.log(num_pos_reviews/(num_pos_reviews + num_neg_reviews))
    priorProb_neg = math.log(num_neg_reviews/(num_pos_reviews + num_neg_reviews))

    #print(priorProb_pos)
    #print(priorProb_neg)

    #for first review determine how many of them are positive
    for review in positivereviews:
        positive_prob = 0
        negative_prob = 0
        for word in review:
            if word in dict_pos_total:
                positive_prob += math.log((dict_pos_total[word] + 1)/(num_pos_words + len(dict_pos_total)))
            if word not in dict_pos_total:
                positive_prob += math.log(1/(num_pos_words + len(dict_pos_total)))
            if word in dict_neg_total:
                negative_prob += math.log((dict_neg_total[word] + 1)/(num_neg_words + len(dict_neg_total)))
            if word not in dict_neg_total:
                negative_prob += math.log(1/(num_neg_words + len(dict_neg_total)))
        positive_prob += priorProb_pos
        negative_prob += priorProb_neg
        if positive_prob > negative_prob:
            num_positive_reviews += 1
        positive_count += 1

    #for the 2nd review determine how many of them are negative
    for review in negativereviews:
        positive_prob = 0
        negative_prob = 0
        for word in review:
            if word in dict_neg_total:
                negative_prob += math.log((dict_neg_total[word] + 1)/(num_neg_words + len(dict_neg_total)))
            if word not in dict_neg_total:
                negative_prob += math.log(1/(num_neg_words + len(dict_neg_total)))
            if word in dict_pos_total:
                positive_prob += math.log((dict_pos_total[word] + 1)/(num_pos_words + len(dict_pos_total)))
            if word not in dict_pos_total:
                positive_prob += math.log(1/(num_pos_words + len(dict_pos_total)))
        positive_prob += priorProb_pos
        negative_prob += priorProb_neg
        if negative_prob > positive_prob:
            num_negative_reviews += 1
        negative_count += 1

    positive_accuracy = float(num_positive_reviews/positive_count)
    negative_accuracy = float(num_negative_reviews/negative_count)
    #print(positive_accuracy)
    #print(negative_accuracy)
    accuracy = (positive_accuracy + negative_accuracy)/2
    return accuracy

def BOWTrain(filepath, filepath2):
    file = open(filepath, 'r', encoding="utf8")
    file2 = open(filepath2, 'r', encoding="utf8")
    reviews = file.read().split("<br /><br />")
    reviews2 = file2.read().split("<br /><br />")
    sentences = []
    file2_sentences = []
    for line in reviews:
        line = re.sub('[^a-zA-Z]+', ' ', line, flags=re.IGNORECASE).lower().split(" ")
        line = [w for w in line if w not in stopwords]
        sentences.append(line)

    for text in reviews2:
        text = re.sub('[^a-zA-Z]+',' ', text, flags=re.IGNORECASE).lower().split(" ")
        text = [word for word in text if word not in stopwords]
        file2_sentences.append(text)

    vocab_file1 = VocabAccumulator(sentences)
    vocab_file2 = VocabAccumulator(file2_sentences)

    bow_dict_pos = dict.fromkeys(vocab_file1,0)
    bow_dict_neg = dict.fromkeys(vocab_file2,0)

    #print(bow_dict_neg)
    #delete empty lists after parsing file
    j = 0
    for i in sentences:
        if len(sentences[j]) == 0:
            del sentences[j]
    k = 0
    for i in file2_sentences:
        if len(file2_sentences[k]) == 0:
            del file2_sentences[k]
    #print (sentences)

    reviews_pos = 0
    reviews_neg = 0
    pos_words = 0
    neg_words = 0
    for word in sentences:
        reviews_pos +=1
        for segment in word:
            if segment in bow_dict_pos:
                bow_dict_pos[segment] += 1
                pos_words += 1

    for word in file2_sentences:
        reviews_neg += 1
        for segment in word:
            if segment in bow_dict_neg:
                bow_dict_neg[segment] += 1
                neg_words += 1
    #print(reviews_pos)
    #print(reviews_neg)
    return bow_dict_pos, bow_dict_neg, reviews_pos, reviews_neg, pos_words, neg_words, sentences, file2_sentences, vocab_file1, vocab_file2

def getTFIDF(posreviews, negreviews):
    num_pos_reviews = len(posreviews) #number of pos reviews
    num_neg_reviews = len(negreviews) #number of neg reviews
    pos_vocab_list = VocabAccumulator(posreviews)
    neg_vocab_list = VocabAccumulator(negreviews)
    pos_TF_dict = dict.fromkeys(pos_vocab_list,0)
    neg_TF_dict = dict.fromkeys(neg_vocab_list, 0)
    pos_IDF_dict = dict.fromkeys(pos_vocab_list,0) #number of reviews that contain word
    neg_IDF_dict = dict.fromkeys(neg_vocab_list,0)

    #compute tf and idf for positive and negative and store in dictionary
    for phrase in posreviews:
        for word in phrase:
            pos_TF_dict[word] += 1
        for word in phrase:
            pos_TF_dict[word] /= len(phrase)
        for i in pos_IDF_dict:
            if i in phrase:
                pos_IDF_dict[i] += 1

    for phrase in negreviews:
        for word in phrase:
            neg_TF_dict[word] += 1
        for word in phrase:
            neg_TF_dict[word] /= len(phrase)
        for i in neg_IDF_dict:
            if i in phrase:
                neg_IDF_dict[i] += 1

    for i in pos_IDF_dict:
        pos_IDF_dict[i] = math.log(float(num_pos_reviews/pos_IDF_dict[i]))
        pos_TF_dict[i] = pos_TF_dict[i] * pos_IDF_dict[i]
    for i in neg_IDF_dict:
        neg_IDF_dict[i] = math.log(float(num_neg_reviews/neg_IDF_dict[i]))
        neg_TF_dict[i] = neg_TF_dict[i] * neg_IDF_dict[i]

    pos_TFIDF_dict = pos_TF_dict
    neg_TFIDF_dict = neg_TF_dict

    return pos_TFIDF_dict, neg_TFIDF_dict

def getGaussianTFIDF(pos_TFIDF_dict, neg_TFIDF_dict, pos_stats, neg_stats):
    pos_prob = 0
    neg_prob = 0
    pos_reviews = 0
    neg_reviews = 0
    positive_count = 0
    negative_count = 0
    pos_stats_dict = {}
    index = 0
    for word, val in pos_TFIDF.items():
        if word in pos_stats and val > 0:
            mean = pos_stats[word][0]
            variance = math.sqrt(pos_stats[word][1])
            if variance != 0:
                exponent = math.exp(-(math.pow(val - mean, 2) / (2 * variance)))
                gauss = (1 / (math.sqrt(2 * math.pi * variance))) * exponent
                if gauss != 0:
                    pos_prob += math.log(gauss)
    pos_stats_dict[index] = pos_prob
    index +=1

    for word, val in neg_TFIDF.items():
        if word in  neg_stats and val > 0:
            mean = neg_stats[word][0]
            variance = math.sqrt(neg_stats[word][1])
            if variance != 0:
                exponent = math.exp(-(math.pow(val - mean, 2) / (2 * variance)))
                gauss = (1 / (math.sqrt(2 * math.pi * variance))) * exponent
                if gauss != 0:
                    neg_prob += math.log(gauss)
    i = 0
    for i in pos_stats_dict:
        if neg_prob < pos_stats_dict[i]:
            pos_reviews +=1
    positive_count +=1
    if neg_prob > pos_prob:
        neg_reviews += 1
    negative_count += 1
    positive_accuracy = float(pos_reviews/positive_count)
    negative_accuracy = float(neg_reviews/negative_count)
    #print(positive_accuracy)
    #print(negative_accuracy)
    accuracy = (positive_accuracy + negative_accuracy)/2

    return accuracy

def getGaussianBOW(PositiveReviews, NegativeReviews, PositiveVocabList, NegativeVocabList, pos_stats, neg_stats):
    pos_prob = 0
    neg_prob = 0
    pos_reviews = 0
    neg_reviews = 0
    positive_count = 0
    negative_count = 0
    pos_bow = dict.fromkeys(PositiveVocabList, 0)
    neg_bow = dict.fromkeys(NegativeVocabList, 0)

    for i in PositiveReviews:
        for word in i:
            if word in pos_bow:
                pos_bow[word] +=1
    for i in NegativeReviews:
        for word in i:
            if word in neg_bow:
                neg_bow[word] +=1

    pos_stats_dict = {}
    for phrase in PositiveReviews:
        index = 0
        for word in phrase:
            if word in pos_stats:
                mean = pos_stats[word][0]
                var = pos_stats[word][1]
                if var != 0:
                #if var == 0: var = .05
                    exponent = math.exp(-(math.pow(pos_bow[word] - mean, 2) / (2 * var)))
                    gauss = (1 / (math.sqrt(2 * math.pi * var))) * exponent
                    if gauss is not 0:
                        pos_prob += gauss
        pos_stats_dict[index] = pos_prob
        index +=1
    for phrase in NegativeReviews:
        for word in phrase:
            if word in neg_stats:
                mean = neg_stats[word][0]
                var = neg_stats[word][1]
                #if var == 0: var = .05
                if var != 0:
                    exponent = math.exp(-(math.pow(neg_bow[word] - mean, 2) / (2 * var)))
                    gauss = (1 / (math.sqrt(2 * math.pi * var))) * exponent
                    if gauss is not 0:
                        neg_prob += gauss
        i = 0
        for i in pos_stats_dict:
            if neg_prob < pos_stats_dict[i]:
                pos_reviews +=1
        positive_count +=1
        if neg_prob > pos_prob:
            neg_reviews += 1
        negative_count += 1

    positive_accuracy = float(pos_reviews/positive_count)
    negative_accuracy = float(neg_reviews/negative_count)
    #print(positive_accuracy)
    #print(negative_accuracy)
    accuracy = (positive_accuracy + negative_accuracy)/2
    return accuracy

def loadStats(posreviews, negreviews, bow_pos_train, bow_neg_train, PosMatrix, NegMatrix):
    pos_stats = dict.fromkeys(bow_pos_train.keys(), (0,0))
    neg_stats = dict.fromkeys(bow_neg_train.keys(), (0,0))
    for phrase in posreviews:
        for word in phrase:
            if word in bow_pos_train:
                pos_mean = (bow_pos_train[word]/len(phrase))
                tuple_list =[pos_mean,0]
                pos_stats[word] = tuple(tuple_list)

    for phrase in negreviews:
        for word in phrase:
            if word in bow_neg_train:
                neg_mean = (bow_neg_train[word]/len(phrase))
                tuple_list =[neg_mean,0]
                neg_stats[word] = tuple(tuple_list)

    for word in pos_stats:
        mean = pos_stats[word][0]
        tuple_list = [mean,0]
        for i in PosMatrix:
            varpos = 0
            if word in i:
                varpos += (i[word] - bow_pos_train[word]) ** 2
            else:
                varpos +=((-1 * mean)**2)
        var = varpos/(len(posreviews)-1)
        tuple_list = [mean,var]
        pos_stats[word] = tuple(tuple_list)

    for word in neg_stats:
        mean = neg_stats[word][0]
        tuple2_list = [mean,0]
        for i in NegMatrix:
            varpos = 0
            if word in i:
                varpos += (i[word] - bow_neg_train[word]) ** 2
            else:
                varpos +=((-1 * mean)**2)
        var = varpos/(len(negreviews)-1)
        tuple2_list = [mean,varpos]
        neg_stats[word] = tuple(tuple2_list)

    return pos_stats, neg_stats


def createSparseMatrix(review):
    temp = []
    for i in review:
        review_dict = {}
        for word in i:
            if word in review_dict:
                 review_dict[word] += 1
            else:
                review_dict[word] = 1
        temp.append(review_dict)
    return temp



##############################################
##############################################
# Start of Main function

PositiveTrainFile = sys.argv[1]
NegativeTrainFile = sys.argv[2]
PositiveTestFile = sys.argv[3]
NegativeTestFile = sys.argv[4]

stopwords = ["in", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now"]
BOW_Train_POS, BOW_Train_NEG, Num_POS_Reviews, Num_NEG_Reviews, POS_Words, NEG_Words, PositiveFileSetences, NegativeFileSentences, PositiveVocabList, NegativeVocabList  = BOWTrain(PositiveTrainFile, NegativeTrainFile)
# print(BOW_Train_POS)
# time.sleep(5)
# print(BOW_Train_NEG)
# time.sleep(5)

# time.sleep(5)
# print(Num_POS_Reviews)
# print(Num_NEG_Reviews)
# TotalWords = POS_Words + NEG_Words
# print(TotalWords)
# print(POS_Words)
# print(len(BOW_Train_POS))
# print(len(BOW_Train_NEG))
# print(NEG_Words)



POS_Reviews, NEG_Reviews = reviewExtraction(PositiveTestFile, NegativeTestFile)

print("BOW Multinomial Accuracy : " + str(getMultinomialBOW(POS_Reviews, NEG_Reviews, BOW_Train_POS, BOW_Train_NEG,POS_Words,NEG_Words, Num_POS_Reviews, Num_NEG_Reviews)))
pos_TFIDF, neg_TFIDF = getTFIDF(PositiveFileSetences, NegativeFileSentences)
#time.sleep(5)
#print(pos_TFIDF)

PosMatrix = createSparseMatrix(POS_Reviews)
#print(PosMatrix)
NegMatrix = createSparseMatrix(NEG_Reviews)

POS_Stats, NEG_Stats = loadStats(POS_Reviews, NEG_Reviews, BOW_Train_POS, BOW_Train_NEG, PosMatrix, NegMatrix)
#print(POS_Stats)
print("BOW Gaussian Accuracy : " + str(getGaussianBOW(POS_Reviews, NEG_Reviews, PositiveVocabList, NegativeVocabList, POS_Stats, NEG_Stats)))
print("TFIDF Gaussian Accuracy : " + str(getGaussianTFIDF(pos_TFIDF,neg_TFIDF,POS_Stats,NEG_Stats)))
