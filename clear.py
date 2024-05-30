import pandas as pd
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

def uploadind_data():
    # Загрузка файлов CSV и создание датафреймов
    raitings_df = pd.read_csv('raitings_df.csv')
    people = pd.read_csv('people.csv')
    brons = pd.read_csv('brons.csv')

    # удалим ненужные строки
    raitings_df = raitings_df[~raitings_df['rubric'].str.contains(
        'Парикмахерская|Барбершоп|Мебель для кухни|Шкафы-купе|Магазин мебели|Тату-салон|Спортивное питание|Фитопродукция, БАДы|Спортивная одежда и обувь|Ювелирный магазин|Ломбард|Магазин часов')]
    return raitings_df, people, brons

def data():
    # Загрузка файлов CSV и создание датафреймов
    top = pd.read_excel('top_10000.xlsx')
    top_m = pd.read_excel('top_m.xlsx')
    restoran_norm = pd.read_excel('restoran_norm.xlsx')
    tmp = pd.read_excel('tmp.xlsx')
    return top, top_m , restoran_norm, tmp

def clean_text(text):
    stop_words = set(stopwords.words('russian'))
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Удаление цифр
    text = re.sub(r'\d+', '', text)

    # Удаление смайлов
    text = re.sub(r':\)|:\(|:-\)', '', text)
    text = re.sub(r'😊', '', text)

    # Удаление мата
    text = re.sub(r'\bмат\b', '', text, flags=re.IGNORECASE)

    # Токенизация текста и удаление стоп-слов
    tokens = nltk.word_tokenize(text.lower())
    cleaned_tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(cleaned_tokens)

def clear_data(raitings_df):
    # Группировка заведений по id и rubric, объединение отзывов
    grouped_data = raitings_df.groupby(['id', 'rubric'])['otziv'].apply(lambda x: ' '.join(x)).reset_index()
    test = grouped_data
    # Применение функции clean_text к каждой ячейке датафрейма с использованием tqdm
    tqdm.pandas(miniters=100)
    test['otziv'] = test['otziv'].progress_apply(clean_text)
    # Стемминг
    stemmer = SnowballStemmer("russian")

    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])

    stemmed_texts_list = []
    for text in tqdm(test['otziv']):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
        text = " ".join(stemmed_tokens)
        stemmed_texts_list.append(text)

    test['text_stem'] = stemmed_texts_list

    print(test.head())

    return test

