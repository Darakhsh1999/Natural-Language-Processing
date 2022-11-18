import random
from nltk.corpus import reuters


def LoadData(n_docs):

    ''' Load in data '''

    # Store document names
    fileids = reuters.fileids() # List of file IDs 
    file_names = [d for d in fileids if d.startswith("test/")] # Filter only the test data
    n_files = len(file_names)
    sampled_files = random.sample(file_names, min(n_docs, n_files))

    docs  = [reuters.raw(doc_id) for doc_id in sampled_files] # List of raw document strings

    return docs
