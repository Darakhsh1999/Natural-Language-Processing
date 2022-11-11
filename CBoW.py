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
from collections import defaultdict
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from CBoWParameters import CBoWParameters
from DocumentPreprocessor import DocumentPreprocessor
from CBoWRepresentation import CBoWRepresentation

# Check data
train_dir = r'Data\brexit_train.tsv'
test_dir = r'Data\brexit_test.tsv'
cond_train = os.path.isfile(train_dir) 
cond_test = os.path.isfile(test_dir)
cond_file = (not cond_train) or (not cond_test) # Check if data is present
if cond_file:
    print("Couldnt detect train and test data.")
    exit(1)
else:
    print("Data files already exist")

# Load in train data
train_corpus = pd.read_csv(train_dir, sep='\t', header=0, names=['label', 'text'])
Xtrain, Xval, Ytrain, Yval = train_test_split(train_corpus.text, train_corpus.label, test_size=0.2, random_state=0)


nlp = spacy.load("en_core_web_sm")

def tokenize(text, nlp, lowercase=True):

    ''' Takes in a document string and tokenizes it
        into word tokens '''

    if lowercase:
        return [t.text.lower() for t in nlp.tokenizer(text)] # Notice difference in spelling (tokenize vs tokenizer), tokenizer is from spaCy
    else:
        return [t.text for t in nlp.tokenizer(text)]

def predict(model, data):   
    
    ''' Forward pass of model '''

    device = next(model.parameters()).device
    outputs = []
    for Xbatch, _ in data:
        Xbatch = Xbatch.to(device)
        scores = model(Xbatch)
        predictions = scores.argmax(dim=1)      
        outputs.append(predictions.detach().cpu().numpy())
    return np.stack(outputs)


def train_model(model, data_train, data_val, par):

    """ Train the model on the given training data.

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
         
        # forward pass part 1: apply the model on X to get 
        # the model's outputs for this batch
        model_output = model(Xbatch)

        # forward pass part 2: compute the loss by comparing
        # the model output to the reference Y values
        loss = loss_func(model_output, Ybatch)
        
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
    

def make_cbow_nn(preprocessor, params):
    # Use a Sequential to build a stacked neural network.
    # We combine the document representation component with a linear output layer.
    return nn.Sequential(
            CBoWRepresentation(preprocessor.voc_size(), params.emb_dim),
            nn.Linear(in_features=params.emb_dim, out_features=preprocessor.n_classes())            
    )


param = CBoWParameters() # Parameters

### Create train and validation data loaders ###

# Set up a prepocessor
rand_state = np.random.randint(0, 100)
Xtrain, Xval, Ytrain, Yval = train_test_split(train_corpus.text, train_corpus.label, test_size=0.2, random_state= rand_state) 
preprocessor = DocumentPreprocessor(tokenizer= tokenize, max_voc_size= param.max_voc, max_len= param.max_len)
preprocessor.build_vocab(Xtrain, Ytrain) # Create vocabulary from train data

# Train data
encoded_train = preprocessor.encode(Xtrain, Ytrain) # encode text to integers
train_data = DataLoader(encoded_train, param.batch_size, shuffle= True, collate_fn= preprocessor.make_batch_tensors) 

# Validation data
encoded_validation = preprocessor.encode(Xval, Yval) # encode text to integers
validation_data = DataLoader(encoded_validation, param.batch_size, shuffle= False, collate_fn= preprocessor.make_batch_tensors)

## Create model ##
model = make_cbow_nn(preprocessor, param)

## Train model ##
if torch.cuda.is_available():
    model.to('cuda')
training_history = train_model(model, train_data, validation_data, param)
model.to('cpu') # After training send device to cpu

# Training history
train_loss = training_history['train_loss']
val_loss = training_history['val_loss']
train_acc = training_history['train_acc']
val_acc = training_history['val_acc']


## Plot training metrics

# Loss
epochs = np.arange(param.n_epochs)
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

## Accuracy
plt.figure(figsize = (15,5)) 
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.legend(["Train", "Validation"])
plt.grid()
plt.show()

## Load in test data ##
test_corpus = pd.read_csv(test_dir, sep='\t', header=0, names=['label', 'text'])
Xtest, Ytest = test_corpus.text, test_corpus.label

# Testa dataloader
encoded_test = preprocessor.encode(Xtest, Ytest) # encode text to integers
test_data = DataLoader(encoded_test, 1, shuffle= False, collate_fn= preprocessor.make_batch_tensors) 

test_predictions = np.squeeze(predict(model, test_data).flatten()) # 0,1

print(classification_report(Ytest, preprocessor.decode_predictions(test_predictions)))
print(accuracy_score(Ytest, preprocessor.decode_predictions(test_predictions)))
