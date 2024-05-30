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


def get_recommendations_ALS(top_m, discription, brons, n_ids=1):

    item = top_m.merge(discription, how='left', on='id')
    item = item.sort_values(by='id')
    item = item.reset_index()
    del item['index']
    del item['Unnamed: 0_y']
    del item['Unnamed: 0_x']
    item = item.reset_index(inplace=False)
    print(item.info())
    print(item.shape)

    user = brons
    user = user.sort_values(by='client_id')

    data1 = {'bron_id': [10000, 10001, 10002, 10003, 10004], 'client_id': [6000, 6000, 6000, 6000, 6000],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [1, 1449, 3, 11372, 2]}
    data2 = {'bron_id': [10005, 10006, 10007, 10008, 10009], 'client_id': [6001, 6001, 6001, 6001, 6001],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [3, 1, 29362, 11370, 28040]}
    data3 = {'bron_id': [10010, 10011, 10012, 10013, 10014], 'client_id': [6002, 6002, 6002, 6002, 6002],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [1, 11372, 28040, 2, 3]}
    data4 = {'bron_id': [10015, 10016, 10017, 10018, 10019], 'client_id': [6003, 6003, 6003, 6003, 6003],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [6103, 16211, 6109, 1, 20627]}
    data5 = {'bron_id': [10020, 10021, 10022, 10023, 10024], 'client_id': [6004, 6004, 6004, 6004, 6004],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [116211, 6109, 27398, 20627, 1]}
    data6 = {'bron_id': [10025, 10026, 10027, 10028, 10029], 'client_id': [6005, 6005, 6005, 6005, 6005],
             'cnt_brons': [5, 5, 5, 5, 5], 'rest_id': [6103, 27397, 6109, 16211, 15902]}

    new_rows1 = pd.DataFrame(data1)
    new_rows2 = pd.DataFrame(data2)
    new_rows3 = pd.DataFrame(data3)
    new_rows4 = pd.DataFrame(data4)
    new_rows5 = pd.DataFrame(data5)
    new_rows6 = pd.DataFrame(data6)

    # Добавление новых строк в датафрейм user
    user = pd.concat([user, new_rows1], ignore_index=True)
    user = pd.concat([user, new_rows2], ignore_index=True)
    user = pd.concat([user, new_rows5], ignore_index=True)
    user = pd.concat([user, new_rows6], ignore_index=True)
    print(user.shape)
    print('user.info()', user.info())
    df = user

    test_user = pd.DataFrame()
    test_user = pd.concat([test_user, new_rows3], ignore_index=True)
    test_user = pd.concat([test_user, new_rows4], ignore_index=True)
    user = user.sort_values(by='client_id')

    # Создание сводной таблицы user-item
    user_item = pd.merge(item[['index', 'id']], user[['client_id', 'rest_id']], left_on='id', right_on='rest_id', how='inner')

    # Группировка по пользователям и товарам
    user_item_grouped = user_item.groupby(['client_id', 'index']).size().reset_index(name='count')

    # Создание sparse matrix
    X_sparse = csr_matrix((user_item_grouped['count'], (user_item_grouped['client_id'], user_item_grouped['index'])))
    print('user_item')
    print(user_item.info())
    print('user_item_g')
    print(user_item_grouped.info())
    print('user_item_g')
    print(user_item_grouped.shape)
    print('gegegegeg')

    model = implicit.nearest_neighbours.CosineRecommender(K=20)  # на основе косинусовой похожести
    model.fit(X_sparse.T)

    #model = implicit.als.AlternatingLeastSquares(factors=16, regularization=0.0, iterations=8)
    #model.fit(X_sparse.T)

    # Пример использования
    user_items = [item[item['id'] == 1]['index'].iloc[0], item[item['id'] == 2]['index'].iloc[0], item[item['id'] == 3]['index'].iloc[0]]
    num_items = 10000
    sparse_matrix_for_user = create_sparse_matrix_for_user(user_items, num_items)
    print(sparse_matrix_for_user)
    print(sparse_matrix_for_user.dtype)

    sparse_matrix_for_user_c = sparse_matrix_for_user.tocsr()
    # Вызов функции recommend() с преобразованной матрицей
    raw_recs = model.recommend(0, sparse_matrix_for_user_c, N=15, filter_already_liked_items=False)

    # Создание датафрейма
    df = pd.DataFrame({'id': raw_recs[0], 'score': raw_recs[1]})
    df = df.merge(item, how='left', left_on='id', right_on='index')
    print(df.info())

    return df
