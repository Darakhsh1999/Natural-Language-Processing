import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
import numpy as np

#nltk.download('reuters')
#nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd


# Note this implementation depends on all initial Z predictions to be 0
class Gibbs():
    def __init__(self, data, n_docs, n_topics, min_df=10, alpha= 0.01, beta =0.01):

        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics
        self.n_docs = n_docs

        # Each word has to occur at least 10 times
        self.vectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=min_df) 
            
        tokenized_docs = self.vectorizer.fit_transform(data)
        self.vocab = np.array(self.vectorizer.get_feature_names_out())
        self.vocab_len = len(self.vocab)

        self.doc_word_counts = tokenized_docs.toarray() # shape (nmb_docs x vocab_len)
        self.word_counts = sum(self.doc_word_counts) # number of occurences of each word across all documents, len = voc_len
        self.words_per_doc = [sum(self.doc_word_counts[i]) for i in range(self.n_docs)] # number of words in each document, len = nmb_docs
        self.total_tokens = sum(self.word_counts)
        self.Z = [np.zeros(s) for s in self.words_per_doc] # topic for every word in every document
        self.word_hash = dict(zip(np.arange(self.vocab_len, self.vocab))) # keys are index and values are word token

        # Row is document, column is topic and value is topic count
        self.topic_freq = np.array([[self.words_per_doc[i]] + [0]*(self.n_topics-1) for i in range(self.n_docs)]) # shape = (nmb_docs x nmb_topics)

        # Row is topic, colum is word and value is number of words with topic k
        self.word_freq = np.array([[self.word_counts[i]] + [0]*(self.n_topics - 1) for i in range(self.vocab_len)]) # shape = (vocab_len x nmb_topics)


    def Run(self, iterations):
        
        ''' Runs iterations of z updates on all worlds '''
        for _ in range(iterations):
            for d_idx in range(self.n_docs):
                for j_idx in range(self.words_per_doc[d_idx]):
                    self.UpdateZ(d_idx, j_idx)


    def UpdateZ(self, d_idx, j_idx):
        # Here we assume fixed d,j
        
        """function to update topic for Z[d_idx][j_idx] according to collapsed Gibbs sampling"""
        old_topic = self.Z[d_idx][j_idx]

        w_idx = self.get_w_idx(d_idx, j_idx)

        P = self.calc_p(d_idx, j_idx, w_idx)

        new_topic = self.sample(P)

        if new_topic != old_topic:
            self.topic_freq[d_idx][old_topic]-=1
            self.topic_freq[d_idx][new_topic]+=1

            self.word_freq[w_idx][old_topic]-=1
            self.word_freq[w_idx][new_topic]+=1

            self.Z[d_idx][j_idx] = new_topic



    def get_w_idx(self,d_idx, j_idx):
        """ Helper function to get correct index for word frequency list"""
        s = 0
        doc_word_count = self.doc_word_counts[d_idx]
        for i in range(self.vocab_len):
            s+= doc_word_count[i]
            if (j_idx + 1) <= s:
                return i

    def calc_p(self,d_idx, j_idx, w_idx):
        """Helper function to calculate P vector"""
        curr_val = self.Z[d_idx][j_idx]

        n_d = self.topic_freq[d_idx]
        
        m_v = self.word_freq[w_idx]

        m = sum(self.word_freq)

        Q = [0]*self.n_topics

        for idx in range(self.n_topics):
            if idx != curr_val:
                Q[idx] = self.calc_q(n_d[idx], m_v[idx], m[idx])
            else: 
                Q[idx] = self.calc_q(n_d[idx]-1, m_v[idx]-1, m[idx]-1)

        q_sum = sum(Q)

        P = [q/q_sum for q in Q]
        return P
        
        
    def calc_q(self, n, m_v, m):
        """Helper function to calculate Q vector"""
        return ((self.alpha + n)*(self.beta + m_v))/(self.beta * self.vocab_len + m)


    def sample(self,P):
        """Helper function to sample new Z_dj topic from given probability distibution P"""
        r = np.random.rand()
        s = 0
        for idx in range(len(P)):
            s+= P[idx]
            if r < s:
                return idx
    




    def simple_eval(self, topic, most_common_x):
        freqs = Counter()

        for d_idx in range(self.nmb_of_docs):
                for j_idx in range(len(self.Z[d_idx])):
                    if self.Z[d_idx][j_idx] == topic:
                        w_idx = self.get_w_idx(d_idx, j_idx)
                        word = self.vocab[w_idx]
                        freqs[word] += 1

        return([word for word,freq in freqs.most_common(most_common_x)])
    


if __name__ == '__main__':
    print("Ran main")