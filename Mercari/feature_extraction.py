import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer

from nltk.corpus import stopwords
import numpy as np

train = pd.read_csv('data/train.tsv', sep='\t')


def fill_missing_data(data):
    data.category_name.fillna(value="Other/Other/Other", inplace=True)
    data.brand_name.fillna(value="Unknown", inplace=True)
    data.item_description.fillna(value="No description yet", inplace=True)
    return data

train = fill_missing_data(train)
# Group name and description fields together as we'll be using them to extract info
train['name_desc'] = train.apply(lambda x: str(x.name).lower() + ' ' + str(x.item_description).lower(), axis=1)

def extract_categories(data):
    data.category_name = data.category_name.fillna('')
    # Create category columns
    cats = pd.DataFrame(data.category_name.str.split('/').tolist(),
                        columns=['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5'])
    # Append to DF
    data = data.join(cats)
    return data

train = extract_categories(train)
train.to_csv('data/cats.csv')


def process_description(data):
    hashing = HashingVectorizer(n_features=500, stop_words='english', alternate_sign=False, norm='l2', binary=False)
    words = hashing.fit_transform(data)
    return words


def group_words(data, stop_words="english"):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=stop_words, max_features=500)
    features = vectorizer.fit_transform(data)
    return features.toarray(), vectorizer.get_feature_names()


def get_word_couts(data):
    bag, words = group_words(data)

    # Sum up the counts of each vocabulary word
    word_count = np.sum(bag, axis=0)
    word_list = sorted(zip(word_count, words), reverse=True)
    return word_list

words_description = get_word_couts(train.item_description)
words_name = get_word_couts(train.name)
words = get_word_couts(train.name_desc)

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for count, word in words[:100]:
#     print count, word

pd.DataFrame(words_name, columns=['count', 'word']).to_csv('data/words_name_stopwords.csv', index=False)
pd.DataFrame(words_description, columns=['count', 'word']).to_csv('data/words_description_stopwords.csv', index=False)
pd.DataFrame(words, columns=['count', 'word']).to_csv('data/word_counts_name_desc.csv', index=False)

size_encoder = {
    'xs': 0,
    'small': 1,
    'sm': 1,
    'medium': 2,
    'l': 3,
    'large': 3,
    'xl': 4
}

gender_encoder = {
    'women': 1,
    'woman': 1,
    'girl': 1,
    'girls': 1,
    'man': 2,
    'men': 2,
    'boy': 2,
    'boys': 2,
    'baby': 3,
    'toddler': 3,
    'tote': 3,
    'kids': 3,
}

# Create on_sale parameter
def create_sale_column(data):
    def is_on_sale(row):
        sale_words = [' sale ', ' off ', ' discount ', ' offer ']
        return int(any(word in row['name_desc'] for word in sale_words))
        # return int(any(word in str(row['name']).lower() for word in sale_words) or any(word in str(row['item_description']).lower() for word in sale_words))

    data['sale'] = data.apply(is_on_sale, axis=1)
    return data

sale_transformer = FunctionTransformer(create_sale_column, validate=False)