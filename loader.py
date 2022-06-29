import torch
from torchtext.legacy.data import Field
import torchtext as tx
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
embedding_size = 100
Train_size = 30000


def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]


def load_data_set(load_my_reviews=False):
    data = pd.read_csv("IMDB Dataset.csv")
    train_data = data[:Train_size]
    train_iter = ReviewDataset(train_data["review"], train_data["sentiment"])
    test_data = data[Train_size:]
    if load_my_reviews:
        my_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
        test_data = test_data.append(my_data)

    test_data = test_data.reset_index(drop=True)
    test_iter = ReviewDataset(test_data["review"], test_data["sentiment"])
    return train_iter, test_iter


embedding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokenize)


def preprocess_review(s):
    cleaned = tokenize(s)
    embedded = embedding.get_vecs_by_tokens(cleaned)
    if embedded.shape[0] != 100 or embedded.shape[1] != 100:
        embedded = torch.nn.functional.pad(embedded, (0, 0, 0, MAX_LENGTH - embedded.shape[0]))
    return torch.unsqueeze(embedded, 0)


def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]


def collect_batch(batch):
    label_list = []
    review_list = []
    embedding_list = []
    for review, label in batch:
        label_list.append(preprocess_label(label))  ### label
        review_list.append(tokenize(review))  ### the  actuall review
        processed_review = preprocess_review(review).detach()
        embedding_list.append(processed_review)  ### the embedding vectors
    label_list = torch.tensor(label_list, dtype=torch.float32).reshape((-1, 2))
    embedding_tensor = torch.cat(embedding_list)
    return label_list.to(device), embedding_tensor.to(device), review_list


##########################
# ADD YOUR OWN TEST TEXT #
##########################

my_test_texts = []
# my_test_texts.append(" this movie is very very bad ,the worst movie ")
# my_test_texts.append(" this movie is so great")
# my_test_texts.append("I really  liked the fish and animations the anther casting was not so good ")
# my_test_labels = ["neg", "pos", "pos"]
my_test_texts.append(
    "This is no walk in the park. I saw this when it came out, and haven't had the guts to watch it again. You " \
    "will never see a more horrifyingly devastating or depressing movie. I felt like I'd been severely beaten. " \
    "What kind of world are we living in when we have children who are treated worse than garbage? This is our " \
    "world, what we have created, what we have allowed to happen. And I would hesitate to say that I-ME-WE are " \
    "not responsible for this. Babenco made this film to wake us up, to shake us to our very")
my_test_texts.append(
    "I love this movie, one of my all time favorites. Ann Blythe as Sally O'Moyne is sweet and trouble-free. She believes that praying to Saint Anne will solve all her and her friends troubles. The sub-plot of the dastardly bad man to get her father's property is funny and clever. Her brothers are what kind of brothers any girl would love to have. Also, look for \"Aunt Bee\" as her mother, a strong Irish woman who won't leave her house that she brought her family up in. They don't make them like this anymore, that's for sure.")
my_test_texts.append(
    "If the creators of this film had made any attempt at introducing reality to the plot, it would have been just one more waste of time, money, and creative effort. Fortunately, by throwing all pretense of reality to the winds, they have created a comedic marvel. Who could pass up a film in which an alien pilot spends the entire film acting like Jack Nicholson, complete with the Lakers T-shirt. Do not dismiss this film as trash.")
my_test_texts.append(
    "i saw this film over 20 years ago and still remember how much i loved it. it really touched me, and i thoroughly enjoyed noel coward's work in it. highly recommended: atmospheric and touching.<br /><br />i think of this film from time to time, and am disappointed it hasn't enjoyed as much of a revival as many classic films. hadn't realized til i searched for it today that it won an academy award for best original story for ben hecht and charles macarthur.<br /><br />basically it involves a nasty character who destroys another's career and is cursed because of it. he dies, but is allowed redemption if he can convince someone to shed a tear over him. the bulk of the movies shows him in pursuit of this goal. well written and lovely. i had known him for his plays so i was surprised to see him in this role on TV late one night in new york. a must see if you ever have the opportunity.")
my_test_labels = ["positive", "positive", "positive", "positive"]


##########################
##########################


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, labels):
        """
        Initialization
        """
        self.labels = labels
        self.reviews = review_list

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y


def get_data_set(batch_size: int, big_batch_size: int, toy: bool = False):
    train_data, test_data = load_data_set(load_my_reviews=toy)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, collate_fn=collect_batch)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True, collate_fn=collect_batch)
    big_train_dataloader = DataLoader(train_data, batch_size=big_batch_size,
                                      shuffle=True, collate_fn=collect_batch)
    big_test_dataloader = DataLoader(test_data, batch_size=big_batch_size,
                                     shuffle=True, collate_fn=collect_batch)
    return train_dataloader, test_dataloader, MAX_LENGTH, embedding_size, big_train_dataloader, \
           big_test_dataloader
