from torch.nn.utils.rnn import pad_sequence


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
import torch
import nltk

DATA_PATH = r"C:\Users\VEREDDAS\Documents\Tomer\resources\LSTM"


def load_data():
    train = pd.read_csv(DATA_PATH + "/snli_1.0/snli_1.0_train.txt", sep="\t")
    dev = pd.read_csv(DATA_PATH + "/snli_1.0/snli_1.0_dev.txt", sep="\t")
    test = pd.read_csv(DATA_PATH + "/snli_1.0/snli_1.0_test.txt", sep="\t")

    # set labels
    train = train[train["gold_label"] != "-"]
    dev = dev[dev["gold_label"] != "-"]

    y_train = train["gold_label"].map(
        {"entailment": 0, "neutral": 1, "contradiction": 2}
    )
    y_dev = dev["gold_label"].map({"entailment": 0, "neutral": 1, "contradiction": 2})

    return (train, y_train), (dev, y_dev), test


def create_embeddings_dict():
    embeddings_dict = {}
    with open(DATA_PATH + "/glove.840B.300d.txt", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parsed_line = line.replace("\n", "").split(" ", 1)
            word = parsed_line[0]
            vector = np.fromstring(parsed_line[1], sep=" ")

            embeddings_dict[word] = vector
    return embeddings_dict


def sen_to_emb(xtrain_seq, embeddings_dict):
    xtrain_seq = xtrain_seq.apply(nltk.word_tokenize)

    embedding_dataset = []
    # create an embedding matrix for the words we have in the dataset
    for row in xtrain_seq:
        embedding_matrix = np.zeros((len(row) + 1, 300))
        for i, word in enumerate(row):

            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_dataset.append(torch.from_numpy(embedding_matrix))

    # zero pad the sequences
    xtrain_pad = pad_sequence(embedding_dataset)
    return xtrain_pad


def df_to_emb(xtrain, embeddings_dict):
    reg_tokenize_fun = lambda text: " ".join(
        nltk.regexp_tokenize(text, pattern=r"-", gaps=True)
    )
    xtrain_seq_premise = xtrain["sentence1"][:1000].apply(reg_tokenize_fun)
    xtrain_seq_hypothesis = xtrain["sentence2"][:1000].apply(reg_tokenize_fun)
    xtrain_seq_premise = sen_to_emb(xtrain_seq_premise, embeddings_dict)
    xtrain_seq_hypothesis = sen_to_emb(xtrain_seq_hypothesis, embeddings_dict)

    return xtrain_seq_premise, xtrain_seq_hypothesis


# print(len(pad_sequence(x, padding_value=-10.0, batch_first=True)[30]))
# padded_df = pad_sequence(x, padding_value=10.0, batch_first=True)


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def create_datasets(xtrain_seq_premise, xtrain_seq_hypothesis, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    padded_df = torch.tensor(len(xtrain_seq_premise), 2, len(xtrain_seq_premise[0]))
    padded_df[0] = xtrain_seq_premise
    padded_df[1] = xtrain_seq_hypothesis
    features_train, features_test, targets_train, targets_test = train_test_split(
        padded_df, target, test_size=0.95, random_state=42
    )

    # Wait, is this a CPU tensor now? Why? Where is .to(device)?
    x_train_tensor = torch.from_numpy(np.array(features_train)).float().to(device)
    y_train_tensor = torch.from_numpy(np.array(targets_train)).float().to(device)

    x_test_tensor = torch.from_numpy(np.array(features_test)).float().to(device)
    y_test_tensor = torch.from_numpy(np.array(targets_test)).float().to(device)

    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])

    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    print(x_test_tensor[0])
    return train_data, test_data
    # should i suffle?


def get_datasets():
    (train, y_train), (dev, y_dev), test = load_data()
    embeddings_dict = create_embeddings_dict()
    xtrain_seq_premise, xtrain_seq_hypothesis = df_to_emb(train, dev, embeddings_dict)
    train_data, test_data = create_datasets(
        xtrain_seq_premise, xtrain_seq_hypothesis, y_train
    )
    return train_data, test_data
