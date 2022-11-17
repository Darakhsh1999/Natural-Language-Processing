import time
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from Auxiliary import LoadData


class Gibbs():
    def __init__(self, data, n_docs, n_topics, min_df= 10, alpha= 0.01, beta= 0.01):

        # Simulation parameters
        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics
        self.n_docs = n_docs

        # Create vocabulary
        self.data = data
        self.vectorizer = CountVectorizer(stop_words= stopwords.words('english'), min_df= min_df) 
        self.analyzer = self.vectorizer.build_analyzer() # seems to be the correct one to use
        tokenized_docs = self.vectorizer.fit_transform(data) # Sparse matrix
        self.vocab = np.array(self.vectorizer.get_feature_names_out())
        self.vocab_len = len(self.vocab)
        self.vocab_hash = dict(zip(self.vocab, np.arange(self.vocab_len))) # keys are word_tokens and values are word_index

        # Token counts and latent variable Z
        self.doc_word_counts = np.array(tokenized_docs.toarray()) # shape (nmb_docs x vocab_len)
        self.word_counts = np.sum(self.doc_word_counts, axis= 0) # number of occurences of each word across all documents, len = voc_len
        self.words_per_doc = np.sum(self.doc_word_counts, axis= 1) # number of words in each document, len = nmb_docs
        self.total_tokens = sum(self.word_counts) # total tokens after processing
        self.Z = [np.random.randint(0, self.n_topics, s) for s in self.words_per_doc] # random topic for every word in every document


        # Topic and Word counts
        self.topic_freq = np.zeros((self.n_docs, self.n_topics))
        self.word_freq = np.zeros((self.vocab_len, self.n_topics))
        self.tokens = self.InitializeTokens() # same shape as Z, but word_index instead of topic
        self.InitializeFrequencies()

        self.SanityCheck()

    def SanityCheck(self):
        ''' Run assertions to check sanity and catch errors '''

        # Check tokenization of document was correct
        for d in range(self.n_docs):
            assert(len(self.tokens[d]) == self.words_per_doc[d])

        # Check sizes of Z and tokens 
        z_size = 0
        w_size = 0
        for d in range(self.n_docs):
            z_size += len(self.Z[d])
            w_size += len(self.tokens[d])
        assert(z_size == w_size)
        assert(z_size == self.total_tokens)

        # Check word count sums
        assert(sum(self.word_counts) == sum(self.words_per_doc))

        # Check token mapping is correct
        random_word_idx = self.tokens[0][1]
        word = self.vocab[random_word_idx]
        assert(random_word_idx == self.vocab_hash.get(word))


    def InitializeTokens(self):
        tokens = []
        for d in range(self.n_docs):
            d_tokens = []
            for token in self.analyzer(self.data[d]):
                if (self.vocab_hash.get(token) != None):
                    d_tokens.append(self.vocab_hash.get(token))
            tokens.append(d_tokens)
            #d_list = [self.vocab_hash.get(token) for token in self.tokenizer(self.data[d]) if (self.vocab_hash.get(token) != None)]
        return tokens  

    def InitializeFrequencies(self):
        
        for d in range(self.n_docs):
            for j in range(self.words_per_doc[d]):

                # topic frequency
                topic_dj = self.Z[d][j]
                self.topic_freq[d][topic_dj] += 1

                # word frequency
                v_idx = self.tokens[d][j]
                self.word_freq[v_idx][topic_dj] += 1


    def Run(self, iterations):
        
        ''' Runs iterations of z updates on all worlds '''
        for _ in range(iterations):
            for d_idx in range(self.n_docs):
                for j_idx in range(self.words_per_doc[d_idx]):
                    self.UpdateZ(d_idx, j_idx)


    def UpdateZ(self, d_idx, j_idx):
        """function to update topic for Z[d_idx][j_idx] according to collapsed Gibbs sampling"""

        old_topic = self.Z[d_idx][j_idx]

        w_idx = self.tokens[d_idx][j_idx]

        P = self.CalculateP(d_idx, j_idx, w_idx)

        new_topic = self.Sample(P)

        if new_topic != old_topic:
            self.topic_freq[d_idx][old_topic]-=1
            self.topic_freq[d_idx][new_topic]+=1

            self.word_freq[w_idx][old_topic]-=1
            self.word_freq[w_idx][new_topic]+=1

            self.Z[d_idx][j_idx] = new_topic


    def CalculateP(self,d_idx, j_idx, w_idx):
        """Helper function to calculate P vector"""

        curr_val = self.Z[d_idx][j_idx]

        n_d = self.topic_freq[d_idx]
        
        m_v = self.word_freq[w_idx]

        m = sum(self.word_freq) # maybe optimze

        Q = [0]*self.n_topics # use numpy array instead

        for idx in range(self.n_topics):
            if idx != curr_val:
                Q[idx] = self.CalculateQ(n_d[idx], m_v[idx], m[idx])
            else: 
                Q[idx] = self.CalculateQ(n_d[idx]-1, m_v[idx]-1, m[idx]-1)

        q_sum = sum(Q)

        P = [q/q_sum for q in Q]
        return P
        
        
    def CalculateQ(self, n, m_v, m):
        """Helper function to calculate Q vector"""
        return ((self.alpha + n)*(self.beta + m_v))/(self.beta * self.vocab_len + m)


    def Sample(self, P):
        """Helper function to sample new Z_dj topic from given probability distibution P"""
        r = np.random.rand()
        s = 0
        for idx in range(self.n_topics):
            s+= P[idx]
            if r < s:
                return idx
    

    def simple_eval(self, topic, most_common_x):
        freqs = Counter()

        for d_idx in range(self.n_docs):
                for j_idx in range(len(self.Z[d_idx])):
                    if self.Z[d_idx][j_idx] == topic:
                        w_idx = self.tokens[d_idx][j_idx]
                        word = self.vocab[w_idx]
                        freqs[word] += 1

        return([word for word,freq in freqs.most_common(most_common_x)])
    


if __name__ == '__main__':
    t_start = time.time()
    n_docs = 1550 
    data = LoadData(n_docs)
    gibbs = Gibbs(data, n_docs, 10, 10, 0.01, 0.01)
    t_end = time.time()
    t_tot = t_end - t_start
    print("Load up time", t_tot)
    t_before = time.time()
    gibbs.Run(1)
    t_after = time.time()
    print("One iterations took", t_after - t_before)