import time
import numpy as np
from tqdm import trange
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from Auxiliary import LoadData


class Gibbs():
    ''' Performs collapsed Gibbs sampling on the LDA posterior distribution'''

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
        self.Q = np.zeros(self.n_topics)

        # Topic and Word counts
        self.topic_freq = np.zeros((self.n_docs, self.n_topics))
        self.word_freq = np.zeros((self.vocab_len, self.n_topics))
        self.tokens = self.InitializeTokens() # same shape as Z, but word_index instead of topic
        self.InitializeFrequencies()
        self.word_freq_sum = np.sum(self.word_freq, axis= 0)

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

        # Word freq checks
        assert(len(self.word_freq_sum) == self.n_topics)

        # Q check
        assert(len(self.Q) == self.n_topics)

    def InitializeTokens(self):
        tokens = []
        for d in range(self.n_docs):
            d_tokens = []
            for token in self.analyzer(self.data[d]):
                if (self.vocab_hash.get(token) != None):
                    d_tokens.append(self.vocab_hash.get(token))
            tokens.append(d_tokens)
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

        for _ in trange(iterations):
            for d_idx in range(self.n_docs):
                for j_idx in range(self.words_per_doc[d_idx]):
                    self.UpdateZ(d_idx, j_idx)


    def UpdateZ(self, d_idx, j_idx):
        """function to update topic for Z[d_idx][j_idx] according to collapsed Gibbs sampling"""

        old_topic = self.Z[d_idx][j_idx]

        w_idx = self.tokens[d_idx][j_idx] # interger value from vocab

        P = self.CalculateP(d_idx, j_idx, w_idx)

        new_topic = self.Sample(P)

        if new_topic != old_topic: # Correct frequency counts

            self.topic_freq[d_idx][old_topic] -= 1
            self.topic_freq[d_idx][new_topic] += 1

            self.word_freq[w_idx][old_topic] -= 1
            self.word_freq[w_idx][new_topic] += 1

            self.word_freq_sum[old_topic] -= 1
            self.word_freq_sum[new_topic] += 1

            self.Z[d_idx][j_idx] = new_topic


    def CalculateP(self,d_idx, j_idx, w_idx):

        curr_val = self.Z[d_idx][j_idx]
        
        # len = n_topics
        n_d = self.topic_freq[d_idx]
        m_v = self.word_freq[w_idx]
        m = self.word_freq_sum 

        for idx in range(self.n_topics):
            if idx != curr_val:
                self.Q[idx] = self.CalculateQ(n_d[idx], m_v[idx], m[idx])
            else: 
                self.Q[idx] = self.CalculateQ(n_d[idx]-1, m_v[idx]-1, m[idx]-1)

        q_sum = sum(self.Q)

        P = (self.Q).copy()/q_sum
        return P
        
        
    def CalculateQ(self, n, m_v, m):
        return ((self.alpha + n)*(self.beta + m_v))/(self.beta * self.vocab_len + m)


    def Sample(self, P):
        ''' Sample new Z_dj topic from given probability distibution P '''

        r = np.random.rand()
        s = 0
        for idx in range(self.n_topics):
            s += P[idx]
            if (r < s): return idx


    def SimpleEval(self, topic, most_common_x):

        freqs = Counter()

        for d_idx in range(self.n_docs):
                for j_idx in range(self.words_per_doc[d_idx]):
                    if self.Z[d_idx][j_idx] == topic:
                        w_idx = self.tokens[d_idx][j_idx]
                        word = self.vocab[w_idx]
                        freqs[word] += 1

        return [word for word,freq in freqs.most_common(most_common_x)]

    def CoherenceScores(self, M= 20):
        
        C = np.zeros(self.n_topics)
        V = []
        D1 = np.zeros((self.n_topics, M))
        D2 = np.zeros((self.n_topics, M, M))

        # List of most common words for each topic, shape (n_topics, M)
        for topic_idx in range(self.n_topics):
            V += [self.SimpleEval(topic_idx, M)]

        # Store top words
        self.top_words = np.array(V)

        # Calculate document and co-document frequency
        for topic_idx in range(self.n_topics):
            for out_idx in range(M):
                
                word_l_idx = self.vocab_hash.get(V[topic_idx][out_idx])
                D1[topic_idx, out_idx] = np.count_nonzero(self.doc_word_counts[:, word_l_idx])

                for in_idx in range(M):

                    word_m_idx = self.vocab_hash.get(V[topic_idx][in_idx])
                    mask_ml = (self.doc_word_counts[:, word_m_idx])*(self.doc_word_counts[:, word_l_idx])
                    D2[topic_idx, out_idx, in_idx] = np.count_nonzero(mask_ml)

        # Calculate coherence scores
        for topic_idx in range(self.n_topics):
            s = 0
            for m in range(1,M):
                for l in range(0, m-1):
                    s += np.log10((1+D2[topic_idx, m, l]) / D1[topic_idx, l])
            C[topic_idx] = s

        return C


if __name__ == '__main__':

    # Parameters
    n_docs = 1550 
    n_topics = 10
    alpha = 0.01
    beta = 0.01
    n_iterations = 100
    M = 20 

    t1 = time.time()
    data = LoadData(n_docs)
    gibbs = Gibbs(data, n_docs, n_topics, min_df= 10, alpha= alpha, beta= beta)
    t2 = time.time()
    gibbs.Run(n_iterations)
    t3 = time.time()
    C = gibbs.CoherenceScores(M= M)
    t4 = time.time()

    print(f"{'Total time':17}: {t4-t1:.3f} s")
    print(f"{'Load data':17}: {t2-t1:.3f} s")
    print(f"{'Run Gibbs':17}: {t3-t2:.3f} s")
    print(f"{'Coherence scores':17}: {t4-t3:.3f} s")

    # Write to simple output file
    sorted_C = np.argsort(-C)
    with open('output_words.txt', 'w') as filehandle:
        filehandle.write(f'Average coherence C = {C.mean()} \n')
        filehandle.write(f'Parameters: K = {n_topics}, alpha = {alpha}, beta = {beta}, iterations = {n_iterations},' +
                         f' documents = {n_docs}, vocab_len = {gibbs.vocab_len}, n_tokens = {gibbs.total_tokens} \n')
        for k in sorted_C:
            filehandle.write(f'C_{k} = {C[k]:.2f}: {gibbs.top_words[k]} \n')
