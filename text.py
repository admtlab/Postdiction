from gensim.models import Word2Vec
import math
import nltk
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from multiprocessing import Pool
from config import config_get
from tqdm import tqdm
import ollama
import hashlib
from pathlib import Path

nltk.download('punkt', quiet=True)

vectorizer = TfidfVectorizer()

zero_threshold = 1e-6



class CorpusIterator:
    def __init__(self, series):
        self.series = series

    def __iter__(self):
        for text in self.series:
            yield nltk.word_tokenize(text)


def get_word2vec_model(data: pd.DataFrame, feature: str, vector_length: int, min_count=1, existing_model=None,
                       save=False) -> tuple:
    """
    Method which generates a word2vec model based off of a string column in a pandas dataframe.

    PARAMETERS:
    data - The input pandas dataframe which contains a column of string data.
    feature - The name of the column which the string data can be extracted.
    min_count - The minimum number of times a word has to appear in the corpus to appear in the word2vec model.
        Since we are going for exact replication, the default value for this parameter is 1.

    RETURN:
    word2vec_model - Python object representing a Word2Vec model for the specified string column.
    """
    corpus = generate_corpus(data, feature)

    if existing_model is None:
        model = Word2Vec(
            vector_size=vector_length,
            min_count=min_count,
            workers=config_get('num_processes')
        )
        model.build_vocab(corpus)
    else:
        model = existing_model
        model.build_vocab(corpus, update=True)

    model.train(corpus,
                total_examples=model.corpus_count,
                epochs=1)

    path = f"models/{feature}_word2vec_model.model"

    return model, path

def _vectorize_single(args):
    text, model, max_length, vector_length = args
    tokens = nltk.word_tokenize(text)
    return np.ravel(text_to_vector(tokens, model, max_length, vector_length))


def vectorize_qwen3(data: pd.DataFrame, feature: str, output_dim: int, chunk_size: int = 10000):
    cache_key = hashlib.md5(pd.util.hash_pandas_object(data[feature]).values).hexdigest()
    cache_path = Path(f".embedding_cache/{feature}_{output_dim}_{cache_key}.npy")
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached embeddings for {feature}")
        embeddings = np.load(cache_path, allow_pickle=False)
        data[feature] = list(embeddings)
        return data

    texts = data[feature].fillna("").astype(str).tolist()
    embeddings = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Embedding with Qwen3"):
        chunk = texts[i:i+chunk_size]
        response = ollama.embed(
            model='qwen3-embedding:0.6b',
            input=chunk,
            dimensions=output_dim,
            keep_alive="60m"
        )
        embeddings.extend(response.embeddings)

    embeddings_array = np.array(embeddings)
    np.save(cache_path, embeddings_array)
    data[feature] = list(embeddings_array)
    return data



def vectorize_column(data: pd.DataFrame, feature: str, word2vec_model, vector_length: int, max_length: int):
    """
    Parallel vectorization of a text column.
    No dependency on pre-tokenized columns.
    Tokenization happens inside each worker.
    """
    # Ensure column can hold arrays
    data[feature] = data[feature].astype(object)

    # Build argument tuples for each row
    args = [
        (text, word2vec_model, max_length, vector_length)
        for text in data[feature].fillna("").astype(str)
    ]

    # Parallel map
    with Pool(config_get("num_processes")) as pool:
        results = pool.map(_vectorize_single, args)

    # Assign results back
    data[feature] = results

    return data, max_length


def _vectorize_single_bin(args: tuple) -> np.ndarray:
    text, max_len = args
    encoded = text.encode("utf-8", errors="backslashreplace")
    arr = np.frombuffer(encoded, dtype=np.uint8)
    padded = np.zeros(max_len, dtype=np.uint8)
    padded[:len(arr)] = arr[:max_len]
    return padded


def vectorize_column_bin(data: pd.DataFrame, feature: str, max_len: int) -> pd.DataFrame:
    handler_method = config_get('handle_nulls')[feature]['method']
    if handler_method == 'outliers':
        data = data.dropna(axis=0, subset=[feature])
    elif handler_method == 'replace_missing_value':
        y_replacement_val = config_get('handle_nulls')[feature]['replacement_value']
        data = data.fillna(value={feature: y_replacement_val})
    else:
        raise ValueError(f"Unknown missing value method: {handler_method}")
    args = [(text, max_len) for text in data[feature]]

    with Pool(config_get("num_processes")) as pool:
        results = pool.map(_vectorize_single_bin, args)

    data[feature] = results

    return data

def stringify_column(data: pd.DataFrame, feature: str, word2vec_model, max_length: int,
                     vector_length: int) -> pd.DataFrame:
    """
    Method which converts a previously vectorized column of a dataframe from a vector back to strings using an input word2vec model.

    PARAMETERS:
    data - The input pandas dataframe which contains a column of vector data.
    feature - The name of the column with the vector data to be converted.
    word2vec_model - The word2vec model which will be used for converting the vectors to strings.
    max_length - The length of each string, that is how many words + padding are in each string.
        This is equal to the length of the longest string in the dataset.

    RETURN:
    data - The output pandas dataframe which contains the column of updated string data.
    """
    for index, row in data.iterrows():
        reshaped_vector = np.array(data.at[index, feature]).reshape((max_length, vector_length))
        data.at[index, feature] = vector_to_text(reshaped_vector, word2vec_model)
    return data


def generate_corpus(data: pd.DataFrame, feature: str) -> CorpusIterator:
    """
        Method which generates and returns a vector of string data, pulled from the feature column of the data df.

        PARAMETERS:
        data - The input pandas dataframe which contains a column of string data.
        feature - The name of the column which contains the string data to build the corpus of.

        RETURN:
        corpus - The vector representation of the feature column from the data df.
        """
    text_series = data[feature].fillna("").astype(str)
    return CorpusIterator(text_series)


def generate_corpus_bin(data: pd.DataFrame, feature: str) -> list:
    corpus = []
    for index, row in data.iterrows():
        if type(row[feature]) != str and math.isnan(row[feature]):
            row[feature] = ""
        corpus.append(row[feature])
    return corpus


def generate_tf_idf_matrix(corpus: list) -> pd.DataFrame:
    X = vectorizer.fit_transform(corpus)
    tfidf_tokens = vectorizer.get_feature_names_out()

    return pd.DataFrame(
        data=X.toarray(),
        index=[f"Index{i}" for i in range(len(corpus))],
        columns=tfidf_tokens
    )


def text_to_vector(text: list[str], word2vec_model, length: int, vector_length: int) -> list:
    """
    Method which converts a line of text to a numerical vector based on the given word2vec_model, adding padding 0 vectors to ensure
    expected length.

    PARAMETERS:
    text - The text to convert from string to a numerical representation.
    word2vec_model - The word2vec model used in the conversion.
    length - The desired length of the outermost vector.
    vector_length - The desired length of each inner vector.

    RETURN:
    vector - A 2D vector which represents the original text in numerical form.
    """
    vector = []
    for i in range(length - len(text)):
        vector.append(np.zeros(vector_length, dtype=np.float32))
    for word in text:
        vector.append(word2vec_model.wv[word])
    return vector


def text_to_vector_bin(text: str) -> int:
    if pd.isnull(text):
        text = ''
    text_enc = text.encode("utf-8", errors="backslashreplace")
    text_num = int.from_bytes(text_enc, byteorder="big", signed=False)

    return text_num


def vector_to_text(vector: list, word2vec_model) -> str:
    """
    Method which converts a numerical representation of text back into its original form, using the given word2vec_model.

    PARAMETERS:
    vector - The 2D vector which represents some text in numerical form, most likely obtained from the text_to_vector method.
    word2vec_model - The word2vec model used in the conversion.

    RETURN:
    text - The text obtained from converting the vector back to its textual representation.
    """
    text = ""
    for arr in vector:
        if np.all(arr < zero_threshold):
            continue
        word_tuple = word2vec_model.wv.most_similar(positive=[arr], topn=1)[0]
        w = word_tuple[0]
        p = word_tuple[1]
        if w[0] in punctuation or (len(w) > 1 and w[1] in punctuation and w[1] != ','):
            text += w
        else:
            text += " " + w
    return text.strip()


def cosine_similarity(input_text_vec: list, generated_text_vec: list) -> float:
    """
    Method for comparing two vectors of equal length. 1 represents highly similar vectors, -1 represents highly dissimilar vectors.
    """
    dot_product = np.dot(input_text_vec, generated_text_vec)
    input_magnitude = np.linalg.norm(input_text_vec)
    generated_magnitude = np.linalg.norm(generated_text_vec)

    denominator = input_magnitude * generated_magnitude
    if denominator == 0 or np.isinf(denominator):
        return 0.0

    cos = dot_product / denominator
    if cos > 1:
        return 1.0
    return cos


def jaccard_similarity(input_text: str, generated_text: str) -> float:
    """
    Method for comparing two pieces of text. 1 represents highly similar text, 0 represents highly dissimilar text.
    """
    input_text_set = set(input_text)
    generated_text_set = set(generated_text)
    intersection = len(input_text_set.intersection(generated_text_set))
    union = len(input_text_set.union(generated_text_set))

    jac = intersection / union
    return jac
