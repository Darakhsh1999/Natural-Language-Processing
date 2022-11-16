from nltk.corpus import reuters
from nltk.corpus import stopwords


def load_data(n_docs):

    ''' Load in data '''

    # Store document names
    fileids = reuters.fileids() # List of file IDs 
    file_names = [d for d in fileids if d.startswith("test/")] # Filter only the test data

    # Store file content as in list of strings
    docs  = [reuters.raw(doc_id) for doc_id in file_names[0:n_docs]] # List of raw document strings

    return docs