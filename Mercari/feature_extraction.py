import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn_pandas import DataFrameMapper

from nltk.corpus import stopwords
import numpy as np
from sklearn.externals import joblib


def fill_missing_data(data):
    data.category_name.fillna(value="Other/Other/Other", inplace=True)
    data.brand_name.fillna(value="Unknown", inplace=True)
    data.item_description.fillna(value="No description yet", inplace=True)
    return data


def pre_process(data):
    # data = fill_missing_data(data)
    data.category_name.fillna(value="Other/Other/Other", inplace=True)
    data.brand_name.fillna(value="Unknown", inplace=True)
    data.item_description.fillna(value="No description yet", inplace=True)
    # group name and description fields together as we'll be using them to extract info
    data['name_desc'] = data.apply(lambda x: str(x['name']).lower() + ' ' + str(x['item_description']).lower(), axis=1)
    # extract categories
    data["cat1"] = data.category_name.str.extract("([^/]+)/[^/]+/[^/]+", expand=False)
    data["cat2"] = data.category_name.str.extract("[^/]+/([^/]+)/[^/]+", expand=False)
    data["cat3"] = data.category_name.str.extract("[^/]+/[^/]+/([^/]+)", expand=False)


def encode_categories(data):
    data["brand_name"] = data.brand_name.astype("category").cat.codes
    data["cat1"] = data.cat1.astype("category").cat.codes
    data["cat2"] = data.cat2.astype("category").cat.codes
    data["cat3"] = data.cat3.astype("category").cat.codes
    return data

def load_data():
    data = pd.read_csv('data/train.tsv', sep='\t')
    data = fill_missing_data(data)
    data['name_desc'] = data.apply(lambda x: str(x['name']).lower() + ' ' + str(x['item_description']).lower(),
                                   axis=1)
    cats = pd.DataFrame(data.category_name.str.split('/').tolist(), columns=['cat1', 'cat2', 'cat3'])
    print cats.head()
    # Append to DF
    data = data.join(cats)
    return data

def extract_categories(data):
    # Create category columns
    cats = pd.DataFrame(data.category_name.str.split('/').tolist(),
                        columns=['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])
    # Append to DF
    data = data.join(cats[['cat1', 'cat2', 'cat3']])
    return data

# def encode_categories(data):
#     encoder1 = LabelEncoder()
#     data.cat1 =


# def process_description(data):
#     hashing = HashingVectorizer(n_features=500, stop_words='english', alternate_sign=False, norm='l2', binary=False)
#     words = hashing.fit_transform(data)
#     return words
#
#
# def group_words(data, stop_words="english"):
#     vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=stop_words, max_features=500)
#     features = vectorizer.fit_transform(data)
#     return features.toarray(), vectorizer.get_feature_names()
#
#
# def get_word_couts(data):
#     bag, words = group_words(data)
#
#     # Sum up the counts of each vocabulary word
#     word_count = np.sum(bag, axis=0)
#     word_list = sorted(zip(word_count, words), reverse=True)
#     return word_list
#
# words_description = get_word_couts(train.item_description)
# words_name = get_word_couts(train.name)
# words = get_word_couts(train.name_desc)
#
# # For each, print the vocabulary word and the number of times it
# # appears in the training set
# # for count, word in words[:100]:
# #     print count, word
#
# pd.DataFrame(words_name, columns=['count', 'word']).to_csv('data/words_name_stopwords.csv', index=False)
# pd.DataFrame(words_description, columns=['count', 'word']).to_csv('data/words_description_stopwords.csv', index=False)
# pd.DataFrame(words, columns=['count', 'word']).to_csv('data/word_counts_name_desc.csv', index=False)


gender_codes = {
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

size_codes = {
        'extra small': 1,
        'extra large': 5,
        ' xs ': 1,
        'small': 2,
        ' sm ': 2,
        'medium': 3,
        ' m ': 3,
        ' l ': 4,
        'large': 4,
        ' xl ': 5,
    }

sale_words = [' sale ', ' off ', ' discount ', ' offer ']

def create_gender_column(data):
    def encode_gender(row):
        for key, value in gender_codes.iteritems():
            if key in row.name_desc:
                return value
        return 0  # Unknown gender
    data['gender'] = data.apply(encode_gender, axis=1)
    return data


def create_size_column(data):
    def encode_size(row):
        for key, value in size_codes.iteritems():
            if key in row.name_desc:
                return value
        return 0  # Unknown size
    data['item_size'] = data.apply(encode_size, axis=1)
    return data


# Create on_sale parameter
def create_sale_column(data):
    def is_on_sale(row):
        return int(any(word in row['name_desc'] for word in sale_words))
        # return int(any(word in str(row['name']).lower() for word in sale_words) or any(word in str(row['item_description']).lower() for word in sale_words))

    data['sale'] = data.apply(is_on_sale, axis=1)
    return data


def filter_columns(data):
    return data[['item_condition_id', 'brand_name', 'shipping', 'item_size', 'gender', 'cat1', 'cat2', 'cat3']]

# def process(row):
#     row['gender'] = encode_gender(row)
#     row['item_size'] = encode_size(row)
#     row['sale'] = is_on_sale(row)
#     return


# def extract_features(data):
#     data = data.apply(process, axis=1)
#     return data

pre_process_transformer = FunctionTransformer(pre_process, validate=False)
# feature_transformer = FunctionTransformer(extract_features, validate=False)
sale_transformer = FunctionTransformer(create_sale_column, validate=False)
size_transformer = FunctionTransformer(create_size_column, validate=False)
gender_transformer = FunctionTransformer(create_gender_column, validate=False)
column_selector = FunctionTransformer(filter_columns, validate=False)
category_encoder = FunctionTransformer(encode_categories, validate=False)

# train_y = train['price']
# train_X = train[['item_condition_id', 'category_name', ]]

# def create_encoders(data):


# mapper = DataFrameMapper([('item_condition_id', None),
#                           ('brand_name', LabelEncoder()),
#                           ('shipping', None),
#                           ('item_size', None),
#                           ('gender', None),
#                           ('cat1', LabelEncoder()),
#                           ('cat2', LabelEncoder()),
#                           ('cat3', LabelEncoder())
#                           ], df_out=True, default=None)

rf = RandomForestRegressor()

if __name__ == '__main__':
    # data = load_data()
    data = pd.read_csv('data/train.tsv', sep='\t', encoding='utf-8')
    y = data['price']
    X = data.drop('price', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

    mapper = DataFrameMapper([('item_condition_id', None),
                              ('brand_name', LabelEncoder()),
                              ('shipping', None),
                              ('item_size', None),
                              ('gender', None),
                              ('cat1', LabelEncoder()),
                              ('cat2', LabelEncoder()),
                              ('cat3', LabelEncoder())
                              ], df_out=True, default=None)

    pipe = make_pipeline(pre_process_transformer, sale_transformer, size_transformer,
                         gender_transformer, column_selector, category_encoder, rf)

    y_train = np.log(1+y_train)
    y_test = np.log(1+y_test)
    pipe.fit(X_train, y_train)

    # joblib.dump(pipe, 'models/rf.pkl')

    y_hat = pipe.predict(X_test.head(5))
    print 'RMSE: {}'.format(pipe.score(X_test.head(5), y_test.head(5)))
