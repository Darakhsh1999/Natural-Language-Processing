import torch
from collections import Counter

class DocumentPreprocessor:
    def __init__(self, tokenizer, max_voc_size=None, max_len=None):
        self.tokenizer = tokenizer
        self.max_voc_size = max_voc_size
        self.max_len = max_len
        self.PAD = "<PAD>"
        self.UNKNOWN = "<UNKNOWN>"

    
    # (1)
    def build_vocab(self, X, Y):
        """
        Build the vocabularies that will be used to encode documents and class labels.

        Parameters: 
          X: a list of document strings.
          Y: a list of document class labels.
        """
        
        # use Counter to collect all unique words with its frequency
        freq = Counter(t for x in X for t in self.tokenizer(x))

        # create a list to store all unique words
        if self.max_voc_size:

            # in case there is a limit of vocabularies
            freq = freq.most_common(self.max_voc_size-2)
            voc = [self.PAD, self.UNKNOWN] + [v[0] for v in freq]
        else:
            voc = [self.PAD, self.UNKNOWN] + list(freq.keys())
        
        # Create X vocabulary by using a dictionary
        self.vocX = {token: id for id, token in enumerate(voc)}

        # Create Y vocabulary by using dictionary
        y = list(set(Y))
        self.vocY = {l: i for i, l in enumerate (y)}

        
    # (2)        
    def n_classes(self):
        """
        Return the number of classes for this classification task.
        """
        return len(self.vocY)

    def voc_size(self):
        """
        Return the number of words in the vocabulary used to encode the document.
        """
        return len(self.vocX)
        
    # (3)        
    def encode(self, X, Y):
        """        
        Carry out integer encoding of a list of documents X and a corresponding list of labels Y.
        
        Parameters: 
          X: a list of document strings.
          Y: a list of class labels.
          
        Returns:
          The list of encoded instances (x, y), where each instance consists of 
          x: list of integer-encoded tokens in the document
          y: integer-encoded class label
        """
        
        # If the user provided a value of the hyperparameter max_len, then any document that is longer than this value needs to be truncated. 
        # For words that are not included in the vocabulary, the special symbol UNKNOWN (hard-coded to index 1) should be used.
        
        # Use a matrix to store encoded tokens per document string
        tokens = []
        for x in X:
          if self.max_len:
            tokens.append([self.vocX.get(t,1) for t in self.tokenizer(x)[:self.max_len] ])
          else:
            tokens.append([self.vocX.get(t,1) for t in self.tokenizer(x)]) 

        # Use a list to store the encoded labels
        labels = [ self.vocY[y] for y in Y]

        return [(t, l) for t, l in zip(tokens, labels)]

    # (4)
    def decode_predictions(self, Y):
        """
        Map a sequence of integer-encoded output labels back to the original labels.

        Parameters: 
          Y: a sequence of integer-encoded class labels.
          
        Returns:
          The sequence of class labels in the original format.
        """
        # create a invert dictionary
        lbl = {v: k for k, v in self.vocY.items()}

        return [lbl[y] for y in Y]    


    # (5)    
    def make_batch_tensors(self, batch):
        """
        Combine a list of instances into two tensors.
        
        Parameters:
          batch: a list of instances (x, y), where each instance is an x-y pair as
                 described for process_data above.
                 
        Returns:
          Two PyTorch tensors Xenc, Yenc, where Xenc contains the integer-encoded documents
          in this batch, and Yenc the integer-encoded labels.
        """
        Xenc =[]
        Yenc =[]
        # get the length of the all documents
        n = max([ len(x) for x, _ in batch] )
        
        for x, y in batch:
          x.extend([0]*(n - len(x)))
          Xenc.append(x)
          Yenc.append(y)

        return torch.as_tensor(Xenc), torch.as_tensor(Yenc)