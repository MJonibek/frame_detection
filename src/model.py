import argparse
import logging
import os
import pathlib

from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import f1_score

from transformers import PreTrainedTokenizerFast, XLNetTokenizerFast

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace, Punctuation
from tokenizers.trainers import WordLevelTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

mispell_dict = {"ain’t": "is not", "aren’t": "are not","can’t": "cannot",
                "’cause": "because", "could’ve": "could have", "couldn’t":
                    "could not", "didn’t": "did not",  "doesn’t": "does not",
                "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not",
                "haven’t": "have not", "he’d": "he would","he’ll": "he will",
                "he’s": "he is", "how’d": "how did", "how’d’y": "how do you",
                "how’ll": "how will", "how’s": "how is",  "I’d": "I would",
                "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have",
                "I’m": "I am", "I’ve": "I have", "i’d": "i would",
                "i’d’ve": "i would have", "i’ll": "i will",
                "i’ll’ve": "i will have","i’m": "i am", "i’ve": "i have",
                "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have",
                "it’ll": "it will", "it’ll’ve": "it will have","it’s": "it is",
                "let’s": "let us", "ma’am": "madam", "mayn’t": "may not",
                "might’ve": "might have","mightn’t": "might not",
                "mightn’t’ve": "might not have", "must’ve": "must have",
                "mustn’t": "must not", "mustn’t’ve": "must not have",
                "needn’t": "need not", "needn’t’ve": "need not have",
                "o’clock": "of the clock", "oughtn’t": "ought not",
                "oughtn’t’ve": "ought not have", "shan’t": "shall not",
                "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                "she’d": "she would", "she’d’ve": "she would have", "she’ll":
                    "she will", "she’ll’ve": "she will have", "she’s": "she is",
                "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have",
                "so’ve": "so have","so’s": "so as", "this’s": "this is","that’d": "that would",
                "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is",
                "they’d": "they would", "they’d’ve": "they would have", "they’ll": "they will",
                "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have",
                "to’ve": "to have", "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have",
                "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are", "we’ve": "we have",
                "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have",
                "what’re": "what are",  "what’s": "what is", "what’ve": "what have", "when’s":
                    "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have",
                "who’s": "who is", "who’ve": "who have", "why’s": "why is", "why’ve": "why have",
                "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have",
                "y’all": "you all", "y’all’d": "you all would","y’all’d’ve": "you all would have",
                "y’all’re": "you all are","y’all’ve": "you all have","you’d": "you would", "you’d’ve": "you would have",
                "you’ll": "you will", "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have", "she`s": "she is", "\n": " "}

def make_dataframe(input_folder, labels_folder=None):
    text = []
    
    for fil in tqdm(filter(lambda x: x.endswith('.txt'),
                           os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(os.path.join(input_folder, fil),
                                              'r', encoding='utf-8').read()
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')
    df = df_text

    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id', 1:'frames'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        df = labels.join(df_text)[['text', 'frames']]

    return df


def read_data(data):
    X_data = data['text'].values
    Y_data = data['frames'].str.split(',').values
    Y_data = encoder.fit_transform(Y_data)

    return (X_data, Y_data)


class SVMBaseline:
    def __init__(self, C=1, kernel='linear', random_state=12345, n_jobs=1):
        self.pipe = Pipeline(
            [('vectorizer', CountVectorizer(ngram_range = (1, 2),
                                             analyzer='word')),
             ('SVM_multiclass', MultiOutputClassifier(
                 svm.SVC(class_weight=None, C=C, kernel=kernel,
                         random_state=random_state),
                 n_jobs=n_jobs))
            ])

    def train(self, X_train, Y_train):
        self.pipe.fit(X_train, Y_train)

    def predict(self, X_test): 
        return self.pipe.predict(X_test)


def replace_typical_misspell(text):
    for key in mispell_dict.keys():
        text = text.replace(key, mispell_dict[key])
    return text


class LRBaseline:
    def __init__(self, C=1, solver='lbfgs', random_state=12345, n_jobs=1):
        self.pipe = Pipeline(
            [('vectorizer', CountVectorizer(ngram_range = (1, 2),
                                            analyzer='word')),
                ('logistic_regression', MultiOutputClassifier(
                 LogisticRegression(class_weight=None, C=C,
                         random_state=random_state, solver=solver, max_iter=500),      #{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
              n_jobs=n_jobs))
             ])

    def train(self, X_train, Y_train):
        self.pipe.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.pipe.predict(X_test)


class SemEvalTask3Subtask2(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=512, labels_1=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.labels_1=labels_1
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        label_1 = self.labels_1[idx]

        encoding = self.tokenizer.encode(
            text,
            padding='max_length',
            max_length=self.max_token_len,
            truncation=True,
            return_tensors='pt'
        )

        return dict(
            input_ids=encoding,
            label=torch.FloatTensor(label),
            label_1=torch.FloatTensor(label_1),
            label_1_output=torch.FloatTensor(label_1)
        )

class CNNClassifier_1(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_size,
                 embedding_size=300,
                 in_channels=1,
                 out_channels=100,
                 kernel_sizes=[3,4,5]):
        super(CNNClassifier_1, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.out_channels,
                       (kernel_size, self.embedding_size))
             for kernel_size in self.kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.out_channels,
                             self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(torch.squeeze(x))  # (batch_size, sequence_length, embedding_size)
        x = x.unsqueeze(1)  # (batch_size, in_channels, sequence_length, embedding_size)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, out_channels, embedding_size), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, out_channels), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes)*out_channels)
        x = self.dropout(x)  # (batch_size, len(kernel_sizes)*out_channels)
        y = self.fc1(x)  # (batch_size, output_size)
        return y

    def predict(self, x, threshold=0.5):
        preds = self.sigmoid(self.forward(x))
        preds = np.array(preds.cpu() > threshold, dtype=float)
        return preds


class CNNClassifier_2(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_size,
                 embedding_size=200,
                 in_channels=1,
                 out_channels=100,
                 kernel_sizes=[3,4,5,3]):
        super(CNNClassifier_2, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.output2=50
        self.embed = nn.Embedding(self.vocab_size, self.embedding_size)
        self.input2 = nn.Linear(6, self.output2)
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.out_channels,
                       (kernel_size, self.embedding_size))
             for kernel_size in self.kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.out_channels+self.output2,
                             self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = self.embed(torch.squeeze(x))  # (batch_size, sequence_length, embedding_size)
        input2 = self.input2(labels)
        x = x.unsqueeze(1)  # (batch_size, in_channels, sequence_length, embedding_size)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, out_channels, embedding_size), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, out_channels), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes)*out_channels)
        x = self.dropout(x)  # (batch_size, len(kernel_sizes)*out_channels)
        combined = torch.cat((x.view(x.size(0), -1),
                              input2.view(input2.size(0), -1)), dim=1)
        y = self.fc1(combined)  # (batch_size, output_size)
        return y

    def predict(self, x, labels, threshold=0.5):
        preds = self.sigmoid(self.forward(x, labels))
        preds = np.array(preds.cpu() > threshold, dtype=float)
        return preds


class ModelType:
    GRU = 1
    LSTM = 2
    CNN_GENERAL = 3
    CNN_MULTY = 4

class AttentionModel:
    NONE = 0
    DOT = 1
    GENERAL = 2

class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))


class Attention(nn.Module):
    def __init__(self, device, method, hidden_size):
        super(Attention, self).__init__()
        self.device = device

        self.method = method
        self.hidden_size = hidden_size

        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.method == AttentionModel.GENERAL:
            self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        batch_size, seq_len, _ = rnn_outputs.shape
        if self.method == AttentionModel.DOT:
            attn_weights = torch.bmm(rnn_outputs, final_hidden_state.unsqueeze(2))
        elif self.method == AttentionModel.GENERAL:
            attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
            attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))

        else:
            raise Exception("[Error] Unknown AttentionModel.")

        attn_weights = torch.softmax(attn_weights.squeeze(2), dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

        return attn_hidden, attn_weights


class RnnClassifier(nn.Module):
    def __init__(self, device, params):
        super(RnnClassifier, self).__init__()
        self.params = params
        self.device = device

        # Embedding layer
        self.word_embeddings = nn.Embedding(self.params.vocab_size, self.params.embed_dim)

        # Calculate number of directions
        self.num_directions = 2 if self.params.bidirectional == True else 1

        self.linear_dims = [self.params.rnn_hidden_dim * self.num_directions] + self.params.linear_dims
        self.linear_dims.append(self.params.label_size)

        # RNN layer
        rnn = None
        if self.params.rnn_type == ModelType.GRU:
            rnn = nn.GRU
        elif self.params.rnn_type == ModelType.LSTM:
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RnnType. Currently supported: RnnType.GRU=1, RnnType.LSTM=2")
        self.rnn = rnn(self.params.embed_dim,
                       self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bidirectional=self.params.bidirectional,
                       dropout=self.params.dropout,
                       batch_first=False)


        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            if self.params.dropout > 0.0:
                self.linears.append(nn.Dropout(p=self.params.dropout))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            self.linears.append(nn.ReLU())

        self.hidden = None

        # Choose attention model
        if self.params.attention_model != AttentionModel.NONE:
            self.attn = Attention(self.device, self.params.attention_model, self.params.rnn_hidden_dim * self.num_directions)
        self.sigmoid = nn.Sigmoid()


    def init_hidden(self, batch_size):
        if self.params.rnn_type == ModelType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif self.params.rnn_type == ModelType.LSTM:
            return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    # def freeze_layer(self, layer):
    #     for param in layer.parameters():
    #         param.requires_grad = False


    def forward(self, inputs):
        batch_size, seq_len, ems = inputs.shape

        # Push through embedding layer
        X = self.word_embeddings(torch.squeeze(inputs)).transpose(0, 1)

        self.hidden = self.init_hidden(batch_size)
        # Push through RNN layer
        rnn_output, self.hidden = self.rnn(X, self.hidden)

        # Extract last hidden state
        final_state = None
        if self.params.rnn_type == ModelType.GRU:
            final_state = self.hidden.view(self.params.num_layers, self.num_directions, batch_size, self.params.rnn_hidden_dim)[-1]
        elif self.params.rnn_type == ModelType.LSTM:
            final_state = self.hidden[0].view(self.params.num_layers, self.num_directions, batch_size, self.params.rnn_hidden_dim)[-1]
        else:
            final_state = self.hidden[0].view(self.params.num_layers, self.num_directions, batch_size, self.params.rnn_hidden_dim)[-1]
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Push through attention layer
        if self.params.attention_model != AttentionModel.NONE:
            rnn_output = rnn_output.permute(1, 0, 2)  #
            X = self.attn(rnn_output, final_hidden_state)[0]
        else:
            X = final_hidden_state

        # Push through linear layers
        for l in self.linears:
            X = l(X)

        return X


    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            # print("Initialize layer with nn.init.xavier_uniform_: {}".format(layer))
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def predict(self, x, threshold=0.5):
        preds = self.sigmoid(self.forward(x))
        preds = np.array(preds.cpu() > threshold, dtype=float)
        return preds


def train_predict_NN(X_train, Y_train, X_test, Y_test, model_type, direction):
    multy_model = True if model_type == ModelType.CNN_MULTY else False
    # train a tokenizer, initialize WordLevel tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    # we first define a normalizer applied before tokenization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    # pre-tokenizer defines a "preprocessing" before the tokenization.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(),
                                                   Digits(individual_digits=True)])
    # training a tokenizer is effectively building a vocabulary in this case
    trainer = WordLevelTrainer(vocab_size=50000, special_tokens=["[PAD]", "[UNK]"])
    tokenizer.train_from_iterator(train_data.text.values, trainer=trainer)
    tokenizer.save("tokenizer.json")
    if multy_model:
        label_first_classifier = range(7)
        label_second_classifier = [label for label in range(14) if label not in label_first_classifier]
    else:
        label_first_classifier = range(14)
        label_second_classifier = label_first_classifier

    #load a tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]"
    )

    seed = 0
    torch.manual_seed(seed)
    train_dataset = SemEvalTask3Subtask2(
        X_train, Y_train[:, label_second_classifier], tokenizer, labels_1=Y_train[:, label_first_classifier]
    )

    test_dataset = SemEvalTask3Subtask2(
        X_test, Y_test[:, label_second_classifier], tokenizer, labels_1=Y_test[:, label_first_classifier]
    )
    #%%
    BATCH_SIZE = 64 # batch size for training

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    parameters_dictionary = {}
    parameters = Parameters({'vocab_size': tokenizer.vocab_size, 'embed_dim': 300,
                         'rnn_hidden_dim': 500, 'bidirectional': direction, 'linear_dims': [128, 300, 14],
                         'label_size': len(encoder.classes_), 'rnn_type': model_type, 'num_layers': 4,
                         'dropout': 0.0, 'attention_model': AttentionModel.GENERAL}
                        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters
    EPOCHS = 30 # epoch
    LR = 0.001  # learning rate
    if model_type == ModelType.CNN_MULTY or model_type == ModelType.CNN_GENERAL:
        model = CNNClassifier_1(
            tokenizer.vocab_size,
            len(label_first_classifier)
        )

    if model_type == ModelType.LSTM or model_type == ModelType.GRU:
        model = RnnClassifier(
            torch.device(device),
            parameters
        )

    model.to(device)

    loss_fun = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        model.train()
        for idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(data['input_ids'].to(device))
            loss = loss_fun(outputs, data['label_1'].to(device))
            loss.backward()
            # print(model.linears[4].weight.grad)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        outputs = []
        targets = []
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):
                output_batch = model.predict(data['input_ids'].to(device))
                target_batch = np.array(data['label_1'])
                outputs.extend(output_batch)
                targets.extend(target_batch)
            # dev_dataloader[idx]['label_1_output'] = outputs

        micro_f1 = f1_score(targets, outputs, average='micro')
        dev_loss = loss_fun(torch.FloatTensor(outputs), torch.FloatTensor(targets))
        print(f'\rEpoch: {epoch}/{EPOCHS}, Micro-f1: {micro_f1:.3f}, Train Loss: {epoch_loss/len(train_dataloader):.3f}, Dev Loss: {dev_loss:.3f}', end='')

    if multy_model:
        test_dataset2 = SemEvalTask3Subtask2(
            X_train, Y_train, tokenizer, labels_1=outputs
        )
        model2 = CNNClassifier_2(
            tokenizer.vocab_size,
            len(label_second_classifier)
        )
        model2.to(device)

        loss_fun = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model2.parameters(), lr=LR)
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0
            model2.train()
            for idx, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model2(data['input_ids'].to(device), data['label_1_output'])
                loss = loss_fun(outputs, data['label'].to(device))
                loss.backward()
                # print(model.linears[4].weight.grad)
                optimizer.step()
                epoch_loss += loss.item()

            model2.eval()
            outputs2 = []
            targets2 = []
            with torch.no_grad():
                for idx, data in enumerate(test_dataset2):
                    output_batch = model2.predict(data['input_ids'].to(device), data['label_1_output'])
                    target_batch = np.array(data['label'])
                    outputs2.extend(output_batch)
                    targets2.extend(target_batch)

            micro_f1 = f1_score(targets2, outputs2, average='micro')
            dev_loss = loss_fun(torch.FloatTensor(outputs), torch.FloatTensor(targets2))
            print(f'\rEpoch: {epoch}/{EPOCHS}, Micro-f1: {micro_f1:.3f}, Train Loss: {epoch_loss/len(train_dataloader):.3f}, Dev Loss: {dev_loss:.3f}', end='')
            permutation_list = label_first_classifier.add(label_second_classifier)
            result = np.array([np.append(a1, b1) for a1, b1 in zip(outputs, outputs2)])
            result = result[:, permutation_list]
    else:
        result=outputs

    return np.array(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train SVM classifier for SemEval-2023 Task 3 - Subtask 2.'
    )
    parser.add_argument('--train',
                        required=True,
                        type=pathlib.Path,
                        help='path to training folder')
    parser.add_argument('--train_label',
                        required=True,
                        type=pathlib.Path,
                        help='path to training label file')
    parser.add_argument('--test',
                        required=True,
                        type=pathlib.Path,
                        help='path to test folder')
    parser.add_argument('--test_label',
                        required=True,
                        type=pathlib.Path,
                        help='path to golden value')
    parser.add_argument('--pred',
                        required=True,
                        type=pathlib.Path,
                        help='path to output prediction')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help='which model to use: "svm", "lr", "cnn", "lstm", "gru", "bilstm", "bigru"')


    seed = 0
    torch.manual_seed(seed)

    args = parser.parse_args()

    # train a tokenizer, initialize WordLevel tokenizer
    # tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    # we first define a normalizer applied before tokenization
    # tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    # # pre-tokenizer defines a "preprocessing" before the tokenization.
    # tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(),
    #                                                Digits(individual_digits=True)])
    # # training a tokenizer is effectively building a vocabulary in this case
    # trainer = WordLevelTrainer(vocab_size=50000, special_tokens=["[PAD]", "[UNK]"])
    # tokenizer.train_from_iterator(train_data.text.values, trainer=trainer)
    # tokenizer.save("tokenizer.json")

    # load a tokenizer
    # tokenizer = PreTrainedTokenizerFast(
    #     tokenizer_file="tokenizer.json",
    #     unk_token="[UNK]",
    #     pad_token="[PAD]"
    # )

    tokenizer = XLNetTokenizerFast(
        tokenizer_file="src/tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]"
    )


    logger.info('Loading data...')
    logger.info(f'Train folder: {args.train.absolute()}')
    logger.info(f'Test folder: {args.test.absolute()}')
    
    train_data = make_dataframe(args.train, args.train_label)
    test_data = make_dataframe(args.test, args.test_label)
    
    X_train = train_data['text'].values
    X_test = test_data['text'].values
    
    encoder = MultiLabelBinarizer()

    Y_train = train_data['frames'].str.split(',').values
    Y_train = encoder.fit_transform(Y_train)

    Y_test = test_data['frames'].str.split(',').values
    Y_test = encoder.fit_transform(Y_test)

    logger.info(f'#train: {len(X_train)}')
    logger.info(f'#test: {len(X_test)}')

    X_train = [replace_typical_misspell(x.lower()) for x in X_train]
    X_test = [replace_typical_misspell(x.lower()) for x in X_test]

    # X_train_tokenized = [tokenizer.encode(
    #     text,
    #     padding='max_length',
    #     max_length=512,
    #     truncation=True,
    #     return_tensors='pt'
    # )[0].numpy() for text in X_train]
    #
    # X_test_tokenized = [tokenizer.encode(
    #     text,
    #     padding='max_length',
    #     max_length=512,
    #     truncation=True,
    #     return_tensors='pt'
    # )[0].numpy() for text in X_test]

    logger.info(f'Training model... {args.model}')
    model = None
    if str(args.model) == 'svm':
        logger.info(f'SVM')
        model = SVMBaseline(C=1, kernel='linear', random_state=12345)
        model.train(X_train, Y_train)
        Y_preds = model.predict(X_test)
    elif str(args.model) == 'lr':
        logger.info(f'LR')
        model = LRBaseline(C=1, solver='lbfgs', random_state=12345)
        model.train(X_train, Y_train)
        Y_preds = model.predict(X_test)
    elif str(args.model) in ['cnn', 'lstm', 'gru', 'multycnn']:
        logger.info(f'CNN MODEL')
        direction = 2 if args.model in ["bilstm", "bigru"] else 1
        if str(args.model) == 'cnn_multy':
            model_type = ModelType.CNN_MULTY
        elif str(args.model) == 'cnn':
            model_type = ModelType.CNN_GENERAL
        elif str(args.model) in ['lstm', 'bilstm']:
            model_type = ModelType.LSTM
        elif str(args.model) in ['gru', 'bigru']:
            model_type = ModelType.GRU
        else:
            model_type = ModelType.CNN_GENERAL
        Y_preds = train_predict_NN(X_train, Y_train, X_test, Y_test, model_type, direction)
    else:
        logger.info(f'Wrong input, train CNN MODEL')
        Y_preds = train_predict_NN(X_train, Y_train, X_test, Y_test, ModelType.CNN_MULTY, 1)

    out = encoder.inverse_transform(Y_preds)
    out = list(map(lambda x: ','.join(x), out))
    out = pd.DataFrame(out, test_data.index)
    out.to_csv(args.pred, sep='\t', header=None)
    logger.info(f'Preditions saved at {args.pred.absolute()}')
