'''
cs 375
tongyu zhou
hw 2
'''
from collections import Counter, namedtuple
import nltk

LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, nminus1grams, ngrams')

def tokenize_text(text, type):
    tokens = []
    for sent in text:
        if nltk.word_tokenize(sent)[0] == type:
            tokens.extend(nltk.word_tokenize(sent)[2:])
    return tokens

# list of ngrams from tokens
def generate_ngrams(tokens, n):
    return [' '.join(tokens[i: i+n]) for i in range(len(tokens) - n+1)]

# builds a language model given the text, n (n-grams), and type (r or p)
def build_lm(text, n, type):
    tokens = tokenize_text(text, type)
    num_tokens = len(tokens)
    vocab = set(tokens)

    n_counts = Counter(generate_ngrams(tokens, n))
    ngrams = {k: v for k, v in n_counts.items() if v > 1}
    n_unknown = {k: v for k, v in n_counts.items() if v == 1}
    ngrams['<UNK>'] = len(n_unknown)

    nminus1_counts = Counter(generate_ngrams(tokens, n-1))
    nminus1grams = {k: v for k, v in nminus1_counts.items() if v > 1}
    nminus1_unknown = {k: v for k, v in nminus1_counts.items() if v == 1}
    nminus1grams['<UNK>'] = len(nminus1_unknown)

    return LanguageModel(num_tokens, vocab, nminus1grams, ngrams)

# calculates katz backoff with absolute discounting for bigrams and unigrams
def katz_backoff(lm, token, history=None):
    n1 = n2 = frequency = 0
    for a, b in lm.ngrams.items():
        if a.split()[0] == history: frequency += 1
        if b == 1: n1 += 1
        if b == 2: n2 += 1
    d = n1 / (n1 + 2*n2)

    if history == None: # unigram model
        ngram_count = lm.ngrams.get(token, 0) if lm.ngrams.get(token, 0) != 0 else lm.ngrams.get('<UNK>', 0)
        suffix_count = lm.ngrams.get(token, 0) if lm.ngrams.get(token, 0) != 0 else lm.ngrams.get('<UNK>', 0)
        prefix_count = lm.num_tokens

        if lm.ngrams.get(token, 0) != 0: return (ngram_count - d) / float(prefix_count)
        else: 
            seen_prob = (frequency / float(prefix_count)) * d
            unseen_prob = prefix_count / float(lm.num_tokens)
            return (seen_prob / unseen_prob) * (suffix_count / lm.num_tokens)

    else: # bigram model
        ngram_count = lm.ngrams.get(history+' '+token, 0) if lm.ngrams.get(history+' '+token, 0) != 0 else lm.ngrams.get('<UNK>', 0)
        suffix_count = lm.nminus1grams.get(token.split()[-1], 0) if lm.nminus1grams.get(token.split()[-1], 0) != 0 else lm.nminus1grams.get('<UNK>', 0)
        prefix_count = lm.nminus1grams.get(history, 0) if lm.nminus1grams.get(history, 0) != 0 else lm.nminus1grams.get('<UNK>', 0)
        
        if lm.ngrams.get(history+' '+token, 0) != 0: return (ngram_count - d) / float(prefix_count)
        else:
            seen_prob = (frequency / float(prefix_count)) * d
            unseen_prob = prefix_count / float(lm.num_tokens)
            return (seen_prob / unseen_prob) * (suffix_count / lm.num_tokens)

# bayes' rule for unigrams
def bayes_classifer(lm, sent):
    prob = 1
    for token in nltk.word_tokenize(sent):
        prob *= (lm.ngrams.get(token,1) / lm.num_tokens)
    return prob

if __name__ == '__main__':
    """ example usage"""

    training = open("hw2_train.txt", 'r').read().splitlines()
    testing = open('hw2_test.txt', 'r').read().splitlines()
    correct = [s[:1] for s in testing]
    
    # uncomment each section to run
    # Katz-backoff bigrams
    '''
    p_lm = build_lm(training, 2, 'p')
    r_lm = build_lm(training, 2, 'r') 
    for sent in testing:
        p_prob = r_prob = 1
        for pair in generate_ngrams(nltk.word_tokenize(sent)[2:], 2):
            history = pair.split()[0]
            token = pair.split()[1]
            p_prob += katz_backoff(p_lm, token, history)
            r_prob += katz_backoff(r_lm, token, history)
        print('p' if p_prob > r_prob else 'r')
    '''
    # Katz-backoff unigrams
    '''
    p_lm = build_lm(training, 1, 'p')
    r_lm = build_lm(training, 1, 'r')
    for sent in testing:
        p_prob = r_prob = 1
        for token in generate_ngrams(nltk.word_tokenize(sent)[2:], 1):
            p_prob += katz_backoff(p_lm, token)
            r_prob += katz_backoff(r_lm, token)
        print('p' if p_prob > r_prob else 'r')
    '''
    # Bayes classification
    '''
    p_lm = build_lm(training, 1, 'p') 
    r_lm = build_lm(training, 1, 'r') 
    e = len(training)
    e_r = e_p = 0
    for sent in training:
        if nltk.word_tokenize(sent)[0] == 'r': e_r += 1 
        else: e_p += 1
    for sent in testing:
        p_prob = (e_r / float(e)) * bayes_classifer(p_lm, sent)
        r_prob = (e_p / float(e)) * bayes_classifer(r_lm, sent)
        print('p' if p_prob > r_prob else 'r')
    '''
    # Correctness checking

    def check(answerkey, prediction, type):
        dict = {}
        tp = fp = fn = 0
        for i, j in zip(answerkey, prediction):
            if i == j and j == type: tp += 1
            if i != j and j == type: fp += 1
            if i != j and j != type: fn += 1
        dict['precision'] = tp / (tp + fp)
        dict['recall'] = tp / (tp + fn)
        dict['f1'] = (2 * dict['precision'] * dict['recall']) / (dict['precision'] + dict['recall'])
        return dict

    katz1 = open("katz1.txt", 'r').read().splitlines()
    katz2 = open("katz2.txt", 'r').read().splitlines()
    bayes = open("bayes.txt", 'r').read().splitlines()
   
    print('Katz-backoff unigram model for p:' + str(check(correct, katz1, 'p')))
    print('Katz-backoff unigram model for r:' + str(check(correct, katz1, 'r')))
    print('Katz-backoff bigram model for p:' + str(check(correct, katz2, 'p')))
    print('Katz-backoff bigram model for r:' + str(check(correct, katz2, 'r')))
    print('Bayes model for p:' + str(check(correct, bayes, 'p')))
    print('Bayes model for r:' + str(check(correct, bayes, 'r')))
    
