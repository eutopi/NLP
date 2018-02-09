import nltk
from nltk.metrics import *
from collections import Counter
import xml.etree.ElementTree as etree

def cooccurrence_features(list, bag):
    features = {b:b in list for b in bag}
    return features

def collocational_features(list):
    features = {}
    for i in range(len(list)): features['pos_' + str(i)] = list[i]
    return features

def parse_xml(tree, window_size, words_list, words_POS_list):
    root = tree.getroot()
    for instance in root[0]:
        for i in instance: 
            if i.tag == 'answer': sense = i.attrib['senseid']
            if i.tag == 'context': 
                POS = [element.attrib['pos'] for element in i]
                words_only = i.text.split()
                words_only.extend([element.tail for element in i])
                words_only.pop(-1)
        words_POS = []
        for a, b in zip(words_only, POS):
            words_POS.append(a)
            words_POS.append(b)
        bank_loc = ['  ' in w for w in words_only]
        words_only = words_only[bank_loc.index(True) - window_size:
                                bank_loc.index(True) + window_size + 1]
        words_POS = words_POS[2*bank_loc.index(True) - 2*window_size:
                              2*bank_loc.index(True) + 2*(window_size + 1)]
        words_only.pop(window_size)
        words_POS = words_POS[:2*window_size] + words_POS[3*window_size:]
        words_list.append((words_only, sense))
        words_POS_list.append((words_POS, sense))

def bag_of_words(list):
    bag = []
    for t in list: bag.extend(t[0])
    return dict(Counter(bag).most_common(10)).keys()

train_tree = etree.parse('bank.n.train.xml')
test_tree = etree.parse('bank.n.test.xml')
window_size = 2
train_words_list, train_words_POS_list, test_words_list, test_words_POS_list = [],[],[],[]
parse_xml(train_tree, window_size, train_words_list, train_words_POS_list)
parse_xml(test_tree, window_size, test_words_list, test_words_POS_list)

words_bag = bag_of_words(train_words_list)
words_POS_bag = bag_of_words(train_words_POS_list)

print('Co-occurrence features with words only')
train_set = [(cooccurrence_features(n, words_bag), sense) for (n, sense) in train_words_list]
test_set = [(cooccurrence_features(n, words_bag), sense) for (n, sense) in test_words_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))  
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)

print('Co-occurrence features with words + POS')
train_set = [(cooccurrence_features(n, words_POS_bag), sense) for (n, sense) in train_words_POS_list]
test_set = [(cooccurrence_features(n, words_POS_bag), sense) for (n, sense) in test_words_POS_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)

print('Collocational features with words only')
train_set = [(collocational_features(n), sense) for (n, sense) in train_words_list]
test_set = [(collocational_features(n), sense) for (n, sense) in test_words_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)

print('Collocational features with words + POS')
train_set = [(collocational_features(n), sense) for (n, sense) in train_words_POS_list]
test_set = [(collocational_features(n), sense) for (n, sense) in test_words_POS_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)

print('Both features with words only')
train_set = [({**collocational_features(n), **cooccurrence_features(n, words_bag)}, sense) 
             for (n, sense) in train_words_list]
test_set = [({**collocational_features(n), **cooccurrence_features(n, words_bag)}, sense) 
            for (n, sense) in test_words_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)

print('Both features with words + POS')
train_set = [({**collocational_features(n), **cooccurrence_features(n, words_bag)}, sense)
             for (n, sense) in train_words_POS_list]
test_set = [({**collocational_features(n), **cooccurrence_features(n, words_bag)}, sense)
            for (n, sense) in test_words_POS_list]
classifier = nltk.NaiveBayesClassifier.train(train_set)
refsets, testsets = [], []
for (list, sense) in test_set:
    refsets.append(sense)
    testsets.append(classifier.classify(list))
print('Accuracy: ' + str(accuracy(refsets, testsets)))
print('Precision: ' + str(precision(set(refsets), set(testsets))))
print('Recall: ' + str(recall(set(refsets), set(testsets))))
print('f1: ' + str(f_measure(set(refsets), set(testsets))))
classifier.show_most_informative_features(5)
