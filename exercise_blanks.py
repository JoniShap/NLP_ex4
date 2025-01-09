import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
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
    return


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
    return


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
        return

    def forward(self, text):
        return

    def predict(self, text):
        return


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
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch_data in data_iter:
            x, y = batch_data
            y_pred = model.predict(x)
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
    negated_accuracy = binary_accuracy(
        test_predictions_tensor[negated_indices_tensor].unsqueeze(1),
        test_labels_tensor[negated_indices_tensor].unsqueeze(1)
    )
    rare_accuracy = binary_accuracy(
        test_predictions_tensor[rare_indices_tensor].unsqueeze(1),
        test_labels_tensor[rare_indices_tensor].unsqueeze(1)
    )

    # Print results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")
    print(f"Negated Polarity Accuracy: {negated_accuracy:.3f}")
    print(f"Rare Words Accuracy: {rare_accuracy:.3f}")

    return train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy

1





def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    return


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    return


import matplotlib.pyplot as plt


def create_training_plots(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Creates and saves two plots:
    1. Loss comparison plot (training vs validation)
    2. Accuracy comparison plot (training vs validation)

    Each plot will help us visualize how the model learns over time and detect potential
    overfitting or underfitting issues.
    """
    # Set up the style for better-looking plots
    plt.style.use('seaborn')

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Training and Validation Loss
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Training and Validation Accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plots
    plt.savefig('w2v_training_plots.png')

    # Display the plots
    plt.show()

if __name__ == '__main__':
    train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy = train_log_linear_with_one_hot()
    print("test_accuracy:", test_accuracy)
    create_training_plots(train_losses, train_accuracies, val_losses, val_accuracies)
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()