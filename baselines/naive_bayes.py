# EECS 487 Intro to NLP
# Assignment 1

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load_ag_news(filename):
    df = pd.read_csv(filename,names=["label","title","description"])
    df["text"] = df.agg(lambda x: f"{x['title']} {x['description']}",axis=1)
    df['label'] = df['label']-1
    # print(df.head())
    return df[['label', 'text']]


def get_basic_stats(df):
    # TODO: calculate mean and std of the number of tokens in the data
    df["tokens"] = df.text.apply(lambda x: word_tokenize(x))
    df["num_tokens"] = df.text.apply(lambda x: len((x)))


    avg_len = df.num_tokens.mean()
    std_len = df.num_tokens.std()
    num_articles = df.groupby('label').count()['text'].to_dict()


    print(f"Average number of token: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of articles in each category: {num_articles}")


class NaiveBayes():
    """Naive Bayes classifier."""

    def __init__(self):
        super().__init__()
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
    
    def fit(self, data):
        self.vectorizer = CountVectorizer(max_df=0.8, min_df=3, ngram_range=(1,2),preprocessor= lambda x: x.lower())
        self.total_count = self.vectorizer.fit_transform(data['text'].to_list()).sum(axis=0)

        self.ngram_count = [
            self.vectorizer.transform(data.loc[data.label == i,"text"].to_list()).sum(axis=0)
            for i in range(4) ] 
        self.category_prob = list(data.groupby('label').count()['text'].to_numpy()/data.shape[0])

            
    
    def calculate_prob(self, docs, c_i, alpha):
        # print(f"calculating probabilities for class {c_i} with alpha {alpha} and {len(docs)} docs")
        word_probs = np.array(self.ngram_count[c_i] + alpha)/(np.sum(self.ngram_count[c_i]) + alpha * len(self.total_count))
        word_probs = np.log(word_probs) 
        
        word_freqs = self.vectorizer.transform(docs)
        prob_sum = word_freqs*word_probs[0]
        prob_sum = prob_sum +  np.log(self.category_prob[c_i])
        return prob_sum

    def predict(self, docs, alpha):
        preds = np.array([self.calculate_prob(docs,i, alpha) for i in range(len(self.category_prob))])
        return preds.argmax(axis=0)


def evaluate(predictions, labels):
    # TODO: calculate accuracy, macro f1, micro f1
    # Note: you can assume labels contain all values from 0 to C - 1, where
    # C is the number of categories

    # accuracy 
    accurate = (pred == label for pred, label in zip(predictions,labels))
    # print(accurate)   
    accuracy = sum(accurate)/len(predictions)

    num_categories = np.max(list(predictions) + list(labels)) + 1
    # print(np.max(predictions), np.max(labels),num_categories)

    # macro_f1 
    tp_sum = 0 
    fp_sum = 0
    fn_sum = 0 
    f1_sum = 0 
    for cat in range(num_categories): 
        tp = sum((pred == cat and label == cat for pred, label in zip(predictions,labels)) )
        fp = sum((pred == cat and label != cat for pred, label in zip(predictions,labels)) )
        fn = sum((pred != cat and label == cat for pred, label in zip(predictions,labels)) )

        f_1 = tp/(tp + ((1/2)* (fp + fn )))
        # print(f"for category {cat}, f1 is {f_1} and  tp, fp, fn = {(tp, fp, fn)}")
        # update macro and micro sums
        tp_sum += tp
        fp_sum += fp 
        fn_sum += fn 
        f1_sum += f_1 

    # print(tp_sum,fp_sum,fn_sum)
    macro_f1 = f1_sum / num_categories
    micro_f1 = tp_sum/(tp_sum + ((1/2)* (fp_sum + fn_sum )))





    return accuracy, macro_f1, micro_f1
