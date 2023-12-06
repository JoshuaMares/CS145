import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
# import operator
# #import seaborn as sns; sns.set()
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 10
# from sklearn.metrics import confusion_matrix

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID]
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. So use smooth methods
        #print(train_data)
        self.Pr_dict = {}
        #N = sum(len(x) for x in train_data)
        for c in range(1,self.num_classes+1):
            cdf = train_data[train_data['classIdx'] == c]#grab all data points whose class is c
            temp_dict = cdf.groupby('wordIdx')['count'].sum().to_dict()#for class c, create dict of count of each word
            #print(temp_dict)
            words_in_class = cdf['count'].sum()#calculate total number of words in class c
            for word in temp_dict:
                temp_dict[word] = (temp_dict[word] + 1) / (words_in_class + self.num_vocab)#prob of seeing word in class c with +1 smoothing
            self.Pr_dict[c] = defaultdict(lambda: (1/(words_in_class + self.num_vocab)), temp_dict)#default dict for any word not seen in class
        # ============================================================
        print("Training completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication to avoid underflow
                ### Remember to add prior probability, i.e. self.pi
                score_dict[classIdx] = np.log(self.pi[classIdx])
                for word in new_dict[docIdx]:
                    score_dict[classIdx] += np.log(self.Pr_dict[classIdx][word])
                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi
            self.pi[c] = 0
            for label in train_label:
                if label == c:
                    self.pi[c] += 1
            self.pi[c] = self.pi[c] / total
            # ============================================================
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))

        print("-"*80)