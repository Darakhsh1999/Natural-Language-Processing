import os.path
import pandas as pd
import spacy
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from spacy.tokens import token
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


cond_train = os.path.isfile(r'Data\brexit_train.tsv') 
cond_test = os.path.isfile(r'Data\brexit_test.tsv')
cond_file = (not cond_train) or (not cond_test) # Download if train or test doesnt exist

# The following shell commands will download the training and test files to your Colab runtime.
if cond_file:
    print("Couldnt detect train and test data.")
    exit(1)
else:
    print("Data files already exist")


train_corpus = pd.read_csv('./brexit_train.tsv', sep='\t', header=0, names=['label', 'text'])

Xtrain, Xval, Ytrain, Yval = train_test_split(train_corpus.text, train_corpus.label, test_size=0.2, random_state=0)


nlp = spacy.load("en_core_web_sm")



def tokenize(text, nlp, lowercase=True):
    if lowercase:
        return [t.text.lower() for t in nlp.tokenizer(text)] # Notice difference in spelling (tokenize vs tokenizer), tokenizer is from spaCy
    else:
        return [t.text for t in nlp.tokenizer(text)]




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
                    

def train_model(model, data_train, data_val, par):
    """Train the model on the given training data.

    Parameters:
      model:      the PyTorch model that will be trained.
      data_train: the DataLoader that generates the input-output batches for training.
      data_val:   the DataLoader for validataion.
      par:        an object containing all relevant hyperparameters.

    Returns:
      history:    a dict containing statistics computed over the epochs.
    """
    
    print("Train device is ", next(model.parameters()).device)
    
    # Define a loss function that is suitable for a multiclass classification task.
    loss_func = nn.CrossEntropyLoss()
    
    # Define an optimizer that will update the model's parameters.
    # You can assume that `par` contains the hyperparameters you need here.
    optimizer = optim.Adam(model.parameters(), lr=par.learning_rate)

    # Contains the statistics that will be returned.
    history = defaultdict(list)

    progress = tqdm(range(par.n_epochs), 'Epochs')        
    for epoch in progress:

        t0 = time.time()

        # Put the model in "training mode". Will affect e.g. dropout, batch normalizers.
        model.train()
        
        # Run the model on the training set, update the model, and get the training set statistics.
        train_loss, train_acc = apply_model(model, data_train, loss_func, optimizer)

        # Put the model in "evaluation mode". Will affect e.g. dropout, batch normalizers.        
        model.eval()
        
        # Turn off gradient computation, since we are not updating the model now.
        with torch.no_grad():
            # Run the model on the validation set and get the training set statistics.
            val_loss, val_acc = apply_model(model, data_val, loss_func)

        t1 = time.time()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['time'].append(t1-t0)
        
        progress.set_postfix({'val_loss': f'{val_loss:.2f}', 'val_acc': f'{val_acc:.2f}'})
    
    return history
        
def apply_model(model, data, loss_func, optimizer=None):
    """Run the neural network for one epoch, using the given batches.
    If an optimizer is provided, this is training data and we will update the model
    after each batch. Otherwise, this is assumed to be validation data.

    Parameters:
      model:     the PyTorch model.
      data:      the DataLoader that generates the input-output batches.
      loss_func: the loss function
      optimizer: the optimizer; should be None if we are running on validation data.

    Returns the loss and accuracy over the epoch."""
    n_correct = 0
    n_instances = 0
    total_loss = 0

    device = next(model.parameters()).device
    

    # for each X, Y pair in the batch:    
    for Xbatch, Ybatch in data:
            
        # put X and Y on the device
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.to(device)
         
        assert(isinstance(Xbatch, torch.Tensor))
        assert(isinstance(Ybatch, torch.Tensor))   
            
        # forward pass part 1: apply the model on X to get 
        # the model's outputs for this batch
        model_output = model(Xbatch)

        assert(len(model_output.shape) == 2)
        assert(model_output.shape[0] == Ybatch.shape[0])
        
        # forward pass part 2: compute the loss by comparing
        # the model output to the reference Y values
        loss = loss_func(model_output, Ybatch)
        
        assert(not loss.shape)
        
        # update the loss statistics
        total_loss += loss.item()

        # convert the scores computed above into hard decisions
        guesses = model_output.argmax(dim=1)
        
        # compute the number of correct predictions and update the statistics
        n_correct += (guesses == Ybatch).sum().item()
        n_instances += Ybatch.shape[0]

        # if we have an optimizer, it means we are processing the training set
        # so that the model needs to be updated after each batch
        if optimizer:
            
            # reset the gradients
            optimizer.zero_grad()
            
            # backprop to compute the new gradients
            loss.backward()
            
            # use the optimizer to update the model
            optimizer.step()
            
    return total_loss/len(data), n_correct/n_instances 
    



class CBoWRepresentation(nn.Module):
    
    def __init__(self, voc_size, emb_dim):
        super().__init__()
        
        # Initialize the parameters. The only parameters of this representation model are the word embeddings.
        self.embedding = nn.Embedding(voc_size, emb_dim)


    def forward(self, X):
        # X is a batch tensor with shape (batch_size, max_doc_length). 
        # Each row contains integer-encoded words.
        
        # Look up the word embeddings for the words in the documents.
        # The result should have the shape (batch_size, max_doc_length, emb_dim)
        embedded = self.embedding(X)
               
        # Compute a mask that hides the padding tokens. We hard-code the padding index 0 here.
        mask = X != 0
        
        # Sum the embeddings for the non-masked positions.
        summed = (embedded.permute((2, 0, 1))*mask).sum(dim=2).t()
        
        # Denominators when computing the means.
        n_not_masked = mask.sum(dim=1, keepdim=True)

        # Compute the means.
        means = summed / n_not_masked
        
        # The result should be a tensor of shape (batch_size, emb_dim)
        return means
    
def make_cbow_nn(preprocessor, params):
    # Use a Sequential to build a stacked neural network.
    # We combine the document representation component with a linear output layer.
    return nn.Sequential(
            CBoWRepresentation(preprocessor.voc_size(), params.emb_dim),
            nn.Linear(in_features=params.emb_dim, out_features=preprocessor.n_classes())            
    )




class CBoWParameters:
    """Container class to store the hyperparameters that control the training process."""

    # Dimensionality of word embeddings.
    emb_dim = 32

    # Learning rate for the optimizer. You might need to change it, depending on which optimizer you use.
    learning_rate = 3e-3

    # Number of training epochs (passes through the training set).
    n_epochs = 20
    
    # Batch size used by the data loaders.
    batch_size = 64

    # Maximum size of vocabulary
    max_voc = 256 

    # Maximum length of document
    max_len = None




param = CBoWParameters() # Parameters


## Create train and validation data loaders ##

# Set up a prepocessor
rand_state = np.random.randint(0, 100)
Xtrain, Xval, Ytrain, Yval = train_test_split(train_corpus.text, train_corpus.label, test_size=0.2, random_state= rand_state) # Lucy: shouldn't we use the pre-splited Train Val data set as the instruction ??????
preprocessor = DocumentPreprocessor(tokenizer=tokenize, max_voc_size= param.max_voc, max_len= param.max_len)

# Create vocabulary from all data to avoid bias 
preprocessor.build_vocab(Xtrain, Ytrain) # create vocabulary

# Train data
encoded_train = preprocessor.encode(Xtrain, Ytrain) # encode text to integers
train_data = DataLoader(encoded_train, param.batch_size, shuffle= True, collate_fn= preprocessor.make_batch_tensors) # Lucy: here shuffling introducing randomness

# Validation data
encoded_validation = preprocessor.encode(Xval, Yval) # encode text to integers
validation_data = DataLoader(encoded_validation, param.batch_size, shuffle= False, collate_fn= preprocessor.make_batch_tensors)

## Create model ##
model = make_cbow_nn(preprocessor, param)

## Train model ##
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    model.to('cuda')
training_history = train_model(model, train_data, validation_data, param)

model.to('cpu')



train_loss = training_history['train_loss']
val_loss = training_history['val_loss']
train_acc = training_history['train_acc']
val_acc = training_history['val_acc']

epochs = np.arange(param.n_epochs)

assert(len(train_loss) == param.n_epochs)

## Loss metrics
plt.figure(figsize = (15,5)) 
plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.xticks(epochs)
plt.legend(["Train", "Validation"])
plt.grid()
plt.title("Training metrics", fontsize = 25)
plt.show()

## Train metrics
plt.figure(figsize = (15,5)) 
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.legend(["Train", "Validation"])
plt.grid()
plt.show()



def predict(model, data):   
    device = next(model.parameters()).device
    outputs = []
    for Xbatch, _ in data:
        Xbatch = Xbatch.to(device)
        scores = model(Xbatch)
        predictions = scores.argmax(dim=1)      
        outputs.append(predictions.detach().cpu().numpy())
    return np.stack(outputs)





## Load in test data ##
test_corpus = pd.read_csv('./brexit_test.tsv', sep='\t', header=0, names=['label', 'text'])
Xtest, Ytest = test_corpus.text, test_corpus.label

# Testa dataloader
encoded_test = preprocessor.encode(Xtest, Ytest) # encode text to integers
test_data = DataLoader(encoded_test, 1, shuffle= False, collate_fn= preprocessor.make_batch_tensors) 

test_predictions = np.squeeze(predict(model, test_data).flatten()) # 0,1

print(classification_report(Ytest, preprocessor.decode_predictions(test_predictions)))
print(accuracy_score(Ytest, preprocessor.decode_predictions(test_predictions)))
