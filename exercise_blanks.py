import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator

import Transformer
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    full_emb = np.zeros(embedding_dim)
    count = 0

    # Sum up embeddings for words that exist in word_to_vec
    for word in sent.text:
        if word in word_to_vec:
            full_emb += word_to_vec[word]
            count += 1

    # If no words were found in word_to_vec, return zero vector
    # Otherwise, return the average
    if count == 0:
        return torch.from_numpy(full_emb).float()
    else:
        return torch.from_numpy(full_emb / count).float()


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    np_one_hot = np.zeros(size)
    np_one_hot[ind] = 1
    return np_one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """

    sum_one_hot = np.zeros(len(word_to_ind))
    for word in sent.text:
        new_vec = get_one_hot(len(word_to_ind), word_to_ind[word])
        sum_one_hot += new_vec
    averaged_vector = sum_one_hot / len(sent.text)
    return torch.from_numpy(averaged_vector).float()

def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    counter = 0
    d = {}
    for word in words_list:
        if word not in d:
            d[word] = counter
            counter += 1
    return d


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    sentence_embedding = np.zeros((seq_len, embedding_dim))

    # Process up to either sentence length or seq_len, whichever is shorter
    for i in range(min(len(sent.text), seq_len)):
        word = sent.text[i]
        if word in word_to_vec:
            sentence_embedding[i] = word_to_vec[word]
        # If word not in word_to_vec, leave as zero vector

    return torch.from_numpy(sentence_embedding).float()

class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        # Initialize the bidirectional LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout layer before the linear classification layer
        self.dropout = nn.Dropout(dropout)

        # Linear layer for classification
        # Input size is hidden_dim*2 because we concatenate forward and backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, text):
        # Get LSTM outputs and final hidden states
        # lstm_out shape: (batch_size, seq_len, hidden_dim*2)
        # hn shape: (2*n_layers, batch_size, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(text)

        # Get the final forward and backward hidden states
        # Reshape hn to separate forward and backward states
        # Each has shape (n_layers, batch_size, hidden_dim)
        forward_hidden = hn[-2, :, :]  # Get last layer's forward hidden state
        backward_hidden = hn[-1, :, :]  # Get last layer's backward hidden state

        # Concatenate forward and backward hidden states
        # Shape: (batch_size, hidden_dim*2)
        concatenated = torch.cat((forward_hidden, backward_hidden), dim=1)

        # Apply dropout
        dropped = self.dropout(concatenated)

        # Project to output space and return logits
        # Shape: (batch_size, 1)
        return self.fc(dropped)

    def predict(self, text):
        # Get logits and apply sigmoid to get probabilities
        return torch.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)


    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    Calculate prediction accuracy for binary classification.

    :param preds: model predictions (logits) with shape [batch_size, 1]
    :param y: true labels with shape [batch_size, 1]
    :return: accuracy as a float between 0 and 1
    """
    # First, apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(preds)

    # Convert probabilities to binary predictions (0 or 1)
    # If probability >= 0.5, predict 1; else predict 0
    y_pred = (probabilities >= 0.5).float()

    # Compare predictions with true labels
    # This creates a tensor of 1s where predictions match labels, 0s where they don't
    correct_predictions = (y_pred == y).float()

    # Calculate accuracy by taking the mean of correct predictions
    accuracy = torch.mean(correct_predictions).item()

    return accuracy


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for batch_data in data_iterator:

        x, y = batch_data
        y = y.unsqueeze(1)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


        total_loss += loss.item() * len(y)
        total_samples += len(y)
        total_acc += binary_accuracy(y_pred, y) *  len(y)

    return total_loss / total_samples , total_acc / total_samples




def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in data_iterator:
            x, y = batch_data
            y = y.unsqueeze(1)

            y_pred = model(x)
            loss = criterion(y_pred, y)


            total_loss += loss.item() * len(y)
            total_samples += len(y)
            total_acc += binary_accuracy(y_pred, y) * len(y)

        return total_loss / total_samples , total_acc / total_samples



def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get model's device
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for x, _ in data_iter:
            # Move input to the same device as model
            x = x.to(device)

            # Get predictions
            y_pred = model.predict(x)

            # Move predictions to CPU and convert to numpy
            y_pred = y_pred.cpu().numpy().flatten()
            all_preds.append(y_pred)

    return np.concatenate(all_preds)



def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    for _ in range(n_epochs):
        train_loss, train_acc = train_epoch(model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion)
        val_loss, val_acc = evaluate(model, data_manager.get_torch_iterator(VAL), criterion)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        print(f"epoch{_+1}/{n_epochs}")
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

    return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, embedding_dim=None)
    vocab_size = data_manager.get_input_shape()[-1]
    model = LogLinear(vocab_size)

    n_epochs = 20
    lr = 0.01
    weight_decay = 0.001

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay
    )

    test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
    test_loss, test_accuracy = evaluate(model, test_iterator, nn.BCEWithLogitsLoss())

    # Get predictions for special subsets
    test_sentences = data_manager.sentences[TEST]

    # Get indices for special subsets
    negated_indices = data_loader.get_negated_polarity_examples(test_sentences)
    rare_indices = data_loader.get_rare_words_examples(test_sentences, data_manager.sentiment_dataset)

    # Get all test predictions
    test_predictions = get_predictions_for_data(model, test_iterator)

    # Convert predictions and labels to tensors
    test_predictions_tensor = torch.tensor(test_predictions)
    test_labels_tensor = torch.tensor(data_manager.get_labels(TEST))

    # Convert indices to tensors to use for indexing
    negated_indices_tensor = torch.tensor(negated_indices)
    rare_indices_tensor = torch.tensor(rare_indices)

    # Calculate accuracy for special subsets using tensors
    negated_predictions = test_predictions_tensor[negated_indices_tensor].unsqueeze(1)
    negated_labels = test_labels_tensor[negated_indices_tensor].unsqueeze(1)
    rare_predictions = test_predictions_tensor[rare_indices_tensor].unsqueeze(1)
    rare_labels = test_labels_tensor[rare_indices_tensor].unsqueeze(1)
    # Calculate accuracy manually for verification
    rare_correct = ((rare_predictions >= 0.5).float() == rare_labels).float().mean()
    negated_correct = ((negated_predictions >= 0.5).float() == negated_labels).float().mean()

    print(f"\nManually calculated rare accuracy: {rare_correct:.3f}")
    print(f"\nManually calculated negated accuracy: {negated_correct:.3f}")


    # Print results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")


    return train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy





def print_sentences(sentences):
    for sentence in sentences:
        print(sentence)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(W2V_EMBEDDING_DIM)
    n_epochs = 20
    lr = 0.01
    weight_decay = 0.001
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay
    )

    test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
    test_loss, test_accuracy = evaluate(model, test_iterator, nn.BCEWithLogitsLoss())

    # Get predictions for special subsets
    test_sentences = data_manager.sentences[TEST]
    print(f"\nTotal number of test sentences: {len(test_sentences)}")

    # Debug test sentences
    print("\nFirst 5 test sentences with their sentiment values:")
    for i, sent in enumerate(test_sentences[:5]):
        print(f"Sentence {i}: {' '.join(sent.text)}")
        print(f"Sentiment value: {sent.sentiment_val}, Sentiment class: {sent.sentiment_class}")

    # Get indices for special subsets
    negated_indices = data_loader.get_negated_polarity_examples(test_sentences)
    rare_indices = data_loader.get_rare_words_examples(test_sentences, data_manager.sentiment_dataset, num_sentences=60)

    print(f"\nNumber of negated polarity examples: {len(negated_indices)}")
    print(f"Number of rare word examples: {len(rare_indices)}")
    print(f"Negated indices range: min={min(negated_indices)}, max={max(negated_indices)}")
    print(f"Rare indices range: min={min(rare_indices)}, max={max(rare_indices)}")

    # Debug rare words selection
    print("\nAnalyzing rare words examples:")
    word_counts = data_manager.sentiment_dataset.get_train_word_counts()
    for i, idx in enumerate(rare_indices[:5]):
        sent = test_sentences[idx]
        print(f"\nRare example {i}:")
        print(f"Sentence: {' '.join(sent.text)}")
        print("Word frequencies in training set:")
        for word in sent.text:
            count = word_counts.get(word, 0)
            print(f"  {word}: {count}")

    # Get all test predictions
    test_predictions = get_predictions_for_data(model, test_iterator)
    print(
        f"\nShape of test predictions: {test_predictions.shape if hasattr(test_predictions, 'shape') else len(test_predictions)}")

    # Convert predictions and labels to tensors
    test_predictions_tensor = torch.tensor(test_predictions)
    test_labels_tensor = torch.tensor(data_manager.get_labels(TEST))

    print(f"Shape of test_predictions_tensor: {test_predictions_tensor.shape}")
    print(f"Shape of test_labels_tensor: {test_labels_tensor.shape}")

    # Debug predictions distribution
    print("\nPredictions statistics:")
    print(f"Mean prediction: {test_predictions_tensor.mean():.3f}")
    print(f"Std prediction: {test_predictions_tensor.std():.3f}")
    print(f"Min prediction: {test_predictions_tensor.min():.3f}")
    print(f"Max prediction: {test_predictions_tensor.max():.3f}")

    # Print prediction counts
    pred_binary = (test_predictions_tensor >= 0.5).float()
    print(f"\nBinary prediction distribution:")
    print(f"Zeros: {(pred_binary == 0).sum()}")
    print(f"Ones: {(pred_binary == 1).sum()}")

    # Convert indices to tensors to use for indexing
    negated_indices_tensor = torch.tensor(negated_indices)
    rare_indices_tensor = torch.tensor(rare_indices)

    # Detailed analysis of rare words predictions
    print("\nDetailed rare words analysis:")
    rare_preds = test_predictions_tensor[rare_indices_tensor]
    rare_labels = test_labels_tensor[rare_indices_tensor]
    rare_binary_preds = (rare_preds >= 0.5).float()
    print(f"Rare predictions mean: {rare_preds.mean():.3f}")
    print(f"Rare predictions distribution:")
    print(f"Zeros: {(rare_binary_preds == 0).sum()}")
    print(f"Ones: {(rare_binary_preds == 1).sum()}")
    print("\nRare words prediction details:")
    for i in range(min(10, len(rare_indices))):
        idx = rare_indices[i]
        pred = test_predictions_tensor[idx]
        label = test_labels_tensor[idx]
        binary_pred = 1 if pred >= 0.5 else 0
        correct = binary_pred == label
        print(f"Example {i}:")
        print(f"  Sentence: {' '.join(test_sentences[idx].text)}")
        print(f"  Raw prediction: {pred:.3f}")
        print(f"  Binary prediction: {binary_pred}")
        print(f"  True label: {label}")
        print(f"  Correct: {correct}")

    # Calculate accuracy for special subsets using tensors
    negated_predictions = test_predictions_tensor[negated_indices_tensor].unsqueeze(1)
    negated_labels = test_labels_tensor[negated_indices_tensor].unsqueeze(1)
    rare_predictions = test_predictions_tensor[rare_indices_tensor].unsqueeze(1)
    rare_labels = test_labels_tensor[rare_indices_tensor].unsqueeze(1)

    # Debug accuracy calculation
    print("\nAccuracy calculation debug:")
    print("Sample of 5 rare predictions vs labels:")
    for i in range(5):
        print(f"Pred: {rare_predictions[i].item():.3f}, Label: {rare_labels[i].item()}")

    # Calculate accuracy manually for verification
    rare_correct = ((rare_predictions >= 0.5).float() == rare_labels).float().mean()
    negated_correct = ((negated_predictions >= 0.5).float() == negated_labels).float().mean()

    print(f"\nManually calculated rare accuracy: {rare_correct:.3f}")
    print(f"\nManually calculated negated accuracy: {negated_correct:.3f}")

    print("\nShapes before accuracy calculation:")
    print(f"Negated predictions shape: {negated_predictions.shape}")
    print(f"Negated labels shape: {negated_labels.shape}")
    print(f"Rare predictions shape: {rare_predictions.shape}")
    print(f"Rare labels shape: {rare_labels.shape}")



    # Print results
    print("\nFinal Results:")
    print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")


    return train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100, n_layers=1, dropout=0.5)
    n_epochs = 4
    lr = 0.001
    weight_decay = 0.0001
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, data_manager, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay
    )

    test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
    test_loss, test_accuracy = evaluate(model, test_iterator, nn.BCEWithLogitsLoss())

    # Get predictions for special subsets
    test_sentences = data_manager.sentences[TEST]

    # Get indices for special subsets
    negated_indices = data_loader.get_negated_polarity_examples(test_sentences)
    rare_indices = data_loader.get_rare_words_examples(test_sentences, data_manager.sentiment_dataset)

    # Get all test predictions
    test_predictions = get_predictions_for_data(model, test_iterator)

    # Convert predictions and labels to tensors
    test_predictions_tensor = torch.tensor(test_predictions)
    test_labels_tensor = torch.tensor(data_manager.get_labels(TEST))

    # Convert indices to tensors to use for indexing
    negated_indices_tensor = torch.tensor(negated_indices)
    rare_indices_tensor = torch.tensor(rare_indices)

    # Calculate accuracy for special subsets using tensors
    negated_predictions = test_predictions_tensor[negated_indices_tensor].unsqueeze(1)
    negated_labels = test_labels_tensor[negated_indices_tensor].unsqueeze(1)
    rare_predictions = test_predictions_tensor[rare_indices_tensor].unsqueeze(1)
    rare_labels = test_labels_tensor[rare_indices_tensor].unsqueeze(1)
    # Calculate accuracy manually for verification
    rare_correct = ((rare_predictions >= 0.5).float() == rare_labels).float().mean()
    negated_correct = ((negated_predictions >= 0.5).float() == negated_labels).float().mean()

    print(f"\nManually calculated rare accuracy: {rare_correct:.3f}")
    print(f"\nManually calculated negated accuracy: {negated_correct:.3f}")



    # Print results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")

    return train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy


import matplotlib.pyplot as plt


def create_training_plots(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Creates and saves two plots:
    1. Loss comparison plot (training vs validation)
    2. Accuracy comparison plot (training vs validation)
    """
    # Import required libraries
    import matplotlib.pyplot as plt

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Training and Validation Loss
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Model Loss over Epochs', fontsize=12, pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=9)

    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Model Accuracy over Epochs', fontsize=12, pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(labelsize=9)

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=2.0)

    # Save the plots with high DPI for better quality
    plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')

    # Display the plots
    plt.show()

    # Close the figure to free memory
    plt.close()

if __name__ == '__main__':
    # Transformer.train_transformer_for_sentiment()
    train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy = train_lstm_with_w2v()
    print("test_accuracy:", test_accuracy)
    create_training_plots(train_losses, train_accuracies, val_losses, val_accuracies)
    # # train_log_linear_with_w2v()
    # train_lstm_with_w2v()