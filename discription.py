# Функция для извлечения наиболее важных и частотных слов
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def get_top_words(text_list):
    vectorizer = CountVectorizer()
    vectorizer.fit(text_list)
    word_freq = vectorizer.transform(text_list)
    sum_words = word_freq.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_words = [word for word, freq in words_freq[:10]]
    return top_words


def frequency_words(test):
    # Применение функции
    tqdm.pandas()
    test['dicr'] = test['text_stem'].progress_apply(lambda x: get_top_words([x]))
    print(test.head())

    discription = pd.DataFrame()
    discription[['id', 'rubric', 'text']] = test[['id', 'rubric', 'dicr']]
    discription.head()

    return discription