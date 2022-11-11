import torch
from torch import nn


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