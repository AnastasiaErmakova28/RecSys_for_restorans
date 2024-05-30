import implicit
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import csr_matrix

import warnings



class ProductEncoder:
    '''
    def __init__(self, product_csv_path):
        self.product_idx = {}
        self.product_pid = {}
        for idx, pid in enumerate(pd.read_csv(product_csv_path).product_id.values):
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid
    '''
    def __init__(self, item):
        self.product_idx = {}
        self.product_pid = {}

        for idx, row in item.iterrows():
            pid = row['product_id']
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.product_idx[pid]
        return [self.product_idx[pid] for pid in x]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.product_pid[idx]
        return [self.product_pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.product_idx)


def make_coo_row(transaction_history, product_encoder: ProductEncoder):
    idx = []
    values = []

    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    n_items = len(items)

    for pid in items:
        idx.append(product_encoder.toIdx(pid))
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    )


def np_normalize_matrix(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norm

def md5_hash(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)


def create_sparse_matrix_for_user(user_items, num_items):
    data = np.ones(len(user_items))  # Создаем массив из единиц длиной равной количеству элементов пользователя
    row_indices = np.zeros(len(user_items))  # Создаем массив нулей для индекса строки
    col_indices = np.array(user_items)  # Индексы столбцов - элементы пользователя

    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(1, num_items))  # Создаем разреженную матрицу формата CSR

    return sparse_matrix

def get_user_items(ids, item):
    user_items = []
    for id in ids:
        index_value = item[item['name_ru'] == id]['index'].iloc[0]
        user_items.append(index_value)
    return user_items


def get_recommendations_ALS(top_m, discription, brons, n_ids):
    # Отключить все предупреждения
    warnings.filterwarnings("ignore")

    item = top_m.merge(discription, how='left', on='id')
    item = item.sort_values(by='id')
    item = item.reset_index()
    del item['index']
    item = item.reset_index(inplace=False)

    user = brons
    user = user.sort_values(by='client_id')

    # Создание сводной таблицы user-item
    user_item = pd.merge(item[['index', 'id']], user[['client_id', 'rest_id']], left_on='id', right_on='rest_id',
                         how='inner')

    # Группировка по пользователям и товарам
    user_item_grouped = user_item.groupby(['client_id', 'index']).size().reset_index(name='count')

    # Создание sparse matrix
    X_sparse = csr_matrix((user_item_grouped['count'], (user_item_grouped['client_id'], user_item_grouped['index'])))

    model = implicit.als.AlternatingLeastSquares(factors=16, regularization=0.0, iterations=8)
    model.fit(X_sparse.T)

    # Ввод n_ids
    n_ids = [str(id) for id in n_ids]  # Преобразование ввода в список целых чисел
    num_items = 10000

    user_items = get_user_items(n_ids, item)
    sparse_matrix_for_user = create_sparse_matrix_for_user(user_items, num_items)

    # Вызов функции recommend() с преобразованной матрицей
    raw_recs = model.recommend(0, sparse_matrix_for_user, N=15, filter_already_liked_items=False)

    # Создание датафрейма
    df = pd.DataFrame({'id': raw_recs[0], 'score': raw_recs[1]})
    df = df.merge(item, how='left', left_on='id', right_on='index')
    df = df[['adress', 'avg_rating', 'cnt_rating', 'rank_score', 'name_ru', 'rubric', 'text']]

    # Включить все предупреждения обратно после выполнения кода
    warnings.filterwarnings("default")

    return df
