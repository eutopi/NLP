'''
cs 375
tongyu zhou
hw 3
'''
import pandas as pd

def transition_counts(corpus):
    tags = []
    for i in corpus: tags.extend([j.split('/')[1] for j in i])    
    # initializes matrix of tags
    temp = list(set(tags))
    temp.append('E')
    table = {i:{j:1 for j in temp} for i in list(set(tags))}
    table['S'] = {j:1 for j in temp}
    # counts occurences of tags in the corpus
    for line in corpus:
        table['S'][line[0].split('/')[1]] += 1
        for first, second in zip(line, line[1:]):
            table[first.split('/')[1]][second.split('/')[1]] += 1
        table[line[-1].split('/')[1]]['E'] += 1
    return table

def emission_counts(corpus):
    tags = []
    words = []
    for i in corpus: tags.extend([j.split('/')[1] for j in i])
    for i in corpus: words.extend([j.split('/')[0] for j in i])
    table = {i:{j:1 for j in list(set(words))} for i in list(set(tags))}
    for line in corpus:
        for l in line:
            table[l.split('/')[1]][l.split('/')[0]] += 1
    return table

def probabilities(table):
    return {i:{j:table[i][j] / float(sum(table[i].values())) for j in table[i].keys()} for i in table.keys()}

def joint_prob(sent, tags, t_prob, e_prob):
    prob = t_prob['S'][tags[0]]
    for i, j in enumerate(tags): prob *= e_prob[tags[i]][sent[i]]
    for first, second in zip(tags, tags[1:]): prob *= t_prob[first][second]
    prob *= t_prob[tags[-1]]['E']
    return prob

def state_prob(word, prev_tag, t_prob, e_prob):
    prob = []
    for k in t_prob[prev_tag].keys():
        if k == 'E': prob.append(float(0))
        else: prob.append(t_prob[prev_tag][k] * e_prob[k][word])
    return prob

################################################################################
text = open("corpus.txt", 'r').read().splitlines()
corpus = [l.split() for l in text]

print('Question 1.1')
print(pd.DataFrame(transition_counts(corpus)))

print('\nQuestion 1.2')
t_prob = probabilities(transition_counts(corpus))
print(pd.DataFrame(t_prob))

print('\nQuestion 1.3')
print(pd.DataFrame(emission_counts(corpus)))

print('\nQuestion 1.4')
e_prob = probabilities(emission_counts(corpus))
print(pd.DataFrame(e_prob))

print('\nQuestion 2.1')
sentence = "show your light when nothing is shining"
a = "NOUN PRON NOUN ADV NOUN VERB NOUN"
b = "VERB PRON NOUN ADV NOUN VERB VERB"
c = "VERB PRON NOUN ADV NOUN VERB NOUN"
print('A prob: ' + str(joint_prob(sentence.split(), a.split(), t_prob, e_prob)))
print('B prob: ' + str(joint_prob(sentence.split(), b.split(), t_prob, e_prob)))
print('C prob: ' + str(joint_prob(sentence.split(), c.split(), t_prob, e_prob)))

print('\nQuestion 2.2')
viterbi_table = []
max_prev = 1 # probability of S
prev_tag = 'S' # sentence start
tags = []
row_names = [k for k in list(t_prob[prev_tag].keys())[:-1]]
for word in sentence.split():
    state = state_prob(word, prev_tag, t_prob, e_prob)
    viterbi_table.append([x * max_prev for x in state[:-1]])
    max_prev = max(viterbi_table[-1])
    max_index = viterbi_table[-1].index(max_prev)
    prev_tag = list(t_prob[prev_tag].keys())[max_index]
    tags.append(prev_tag)
table = pd.DataFrame(viterbi_table, index = sentence.split(), columns = row_names).transpose()
print(table)
print('POS tagging :' + str(tags))
print('Probability: ' + str(max(viterbi_table[-1]) * t_prob[tags[-1]]['E']))
