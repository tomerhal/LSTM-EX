from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple
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


# def create_embeddings_dict():
#     embeddings_dict = {}
#     with open(DATA_PATH + "/glove.840B.300d.txt", "r", encoding="utf-8") as f:
#         for line in tqdm(f):
#             parsed_line = line.replace("\n", "").split(" ", 1)
#             word = parsed_line[0]
#             vector = np.fromstring(parsed_line[1], sep=" ")

#             embeddings_dict[word] = vector
#     return embeddings_dict

"""
1. prefer comprehension and generators over loops
2. separate iteration logic over single line processing logic -- improves readability
3. some naming improvements
"""

def create_embeddings_dict(path_to_glove_embeddings: str) -> Dict[str, np.ndarray]:
    with open(path_to_glove_embeddings, encoding="utf-8") as embeddings_file:
        return dict(parse_embedding_line(line) for line in tqdm(embeddings_file))


def parse_embedding_line(embedding_line: str) -> Tuple[str, np.ndarray]:
    parsed_line = embedding_line.replace("\n", "")
    word, embedding_string = parsed_line.split(" ", 1)
    embedding = np.fromstring(embedding_string, sep=" ")

    return word, embedding



# def sen_to_emb(xtrain_seq, embeddings_dict):
#     xtrain_seq = xtrain_seq.apply(nltk.word_tokenize)

#     embedding_dataset = []
#     # create an embedding matrix for the words we have in the dataset
#     for row in xtrain_seq:
#         embedding_matrix = np.zeros((len(row) + 1, 300))
#         for i, word in enumerate(row):

#             embedding_vector = embeddings_dict.get(word)
#             if embedding_vector is not None:
#                 embedding_matrix[i] = embedding_vector

#         embedding_dataset.append(torch.from_numpy(embedding_matrix))

#     # zero pad the sequences
#     xtrain_pad = pad_sequence(embedding_dataset)
#     return xtrain_pad

"""
    1. separate looping logic from single word / sentence processing -> greatly simplifies code, improves readability
"""

EMBEDDING_SIZE = 300 # should be in a constants file / parameter

def get_padded_embeddings(sentences: pd.Series, embeddings_dict: Dict[str, np.ndarray]) -> torch.Tensor:
    sentence_embeddings = sentences.apply(embed_sentence) # maybe need to convert to tuple
    return pad_sequence(torch.stack(sentence_embeddings, embeddings_dict))

def embed_sentence(sentence: str, embeddings_dict: Dict[str, np.ndarray]) -> torch.Tensor:
    words = nltk.word_tokenize(sentence)
    word_embeddings = (embeddings_dict.get(word, default=np.zeros(EMBEDDING_SIZE)) for word in words)
    return torch.from_numpy(np.concatenate(word_embeddings))



"""
even better: a class based implementation, and then we don't need to carry the embedding_dict around
"""

DEFAULT_EMBEDDING_SIZE = 300

TokenizerType = Callable[[str], Sequence[str]]

@dataclass
class Embedder:
    embedding_dict: Dict[str, np.ndarray]

    embedding_size: int = DEFAULT_EMBEDDING_SIZE
    tokenizer: TokenizerType = nltk.word_tokenize

    def get_padded_embeddings(self, sentences: pd.Series) -> torch.Tensor:
        sentence_embeddings = sentences.apply(embed_sentence) # maybe need to convert to tuple
        return pad_sequence(torch.stack(sentence_embeddings)) # notice -> no extra embedding_dict parameter here
    
    def embed_sentence(self, sentence: str) -> torch.Tensor: # no extra parameter
        words = self.tokenizer(sentence)
        word_embeddings = (self.embed_word(word) for word in words)
        return torch.from_numpy(np.concatenate(word_embeddings))
    
    def embed_word(self, word: str) -> torch.Tensor:
        # even better - create embeddings_dict with torch tensors from the start, to save the numpy->torch conversion
        return torch.from_numpy(self.embeddings_dict.get(word, default=np.zeros(self.embedding_size)))

"""
Then usage is also simplified:

embedder = Embedder(embedding_dict)
xtrain_seq_premise = embedder(xtrain_seq_premise)
xtrain_seq_hypothesis = embedder(xtrain_seq_hypothesis)

"""





def df_to_emb(xtrain, embeddings_dict):
    """
    simplify to str.replace, and should be in a separate preprocessing function/module.
    Then - this function is kind of redundante since all it does is apply get_padded_embeddings
    """
    reg_tokenize_fun = lambda text: " ".join(
        nltk.regexp_tokenize(text, pattern=r"-", gaps=True)
    )

    
    xtrain_seq_premise = xtrain["sentence1"][:1000].apply(reg_tokenize_fun)
    xtrain_seq_hypothesis = xtrain["sentence2"][:1000].apply(reg_tokenize_fun)
    xtrain_seq_premise = get_padded_embeddings(xtrain_seq_premise, embeddings_dict)
    xtrain_seq_hypothesis = get_padded_embeddings(xtrain_seq_hypothesis, embeddings_dict)

    return xtrain_seq_premise, xtrain_seq_hypothesis


# print(len(pad_sequence(x, padding_value=-10.0, batch_first=True)[30]))
# padded_df = pad_sequence(x, padding_value=10.0, batch_first=True)


"""
No need for this
"""
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
