
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

