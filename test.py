import datetime
import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
import rectools

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel
from rectools.models import PopularModel
from rectools.models import PureSVDModel
from rectools.models import ImplicitALSWrapperModel
from implicit.als import AlternatingLeastSquares
import warnings

def start(brons, user, restoran_norm):
    rating = brons.copy()
    rating = user.merge(restoran_norm, how='left', left_on='rest_id', right_on='id')
    rating = rating[['client_id', 'rest_id', 'rank_score']]
    rating = rating.rename(columns={'client_id': 'user_id', 'rank_score': 'weight', 'rest_id': 'item_id'})
    # Поменяем порядок столбцов
    rating = rating.reindex(columns=['user_id', 'item_id', 'weight'])
    rating = rating.sort_values(by='user_id')
    rating['datetime'] = datetime.datetime.now()
    rating = rating.fillna(0)

    # Prepare a dataset to build a model
    dataset = Dataset.construct(rating)
    return dataset, rating

def top(brons, user, restoran_norm, tmp):
    dataset, ratings = start(brons, user, restoran_norm)
    # Fit model and generate recommendations for all users

    model = PopularModel()
    model.fit(dataset)
    recos = model.recommend(
        users=ratings[Columns.User].unique(),
        dataset=dataset,
        k=10,
        filter_viewed=True,
    )

    recos = recos.merge(tmp, how='left', left_on='item_id', right_on='id')
    return recos

def ALS(brons, user, restoran_norm, tmp):
    # Отключить все предупреждения
    warnings.filterwarnings("ignore")
    dataset, ratings = start(brons, user, restoran_norm)
    # Fit model and generate recommendations for all users
    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=64,
            regularization=0.01,
            alpha=1,
            random_state=2023,
            use_gpu=False,
            iterations=15))
    model.fit(dataset)
    recos = model.recommend(
        users=ratings[Columns.User].unique(),
        dataset=dataset,
        k=10,
        filter_viewed=True,
    )
    recos = recos.merge(tmp, how='left', left_on='item_id', right_on='id')
    warnings.filterwarnings("default")
    return recos



def item_to_item(brons, user, restoran_norm, tmp):
    # Отключить все предупреждения
    warnings.filterwarnings("ignore")
    dataset, ratings = start(brons, user, restoran_norm)
    model = ImplicitItemKNNWrapperModel(
        model=TFIDFRecommender(K=5)
    )
    model.fit(dataset)
    recos = model.recommend(
        users=ratings[Columns.User].unique(),
        dataset=dataset,
        k=10,
        filter_viewed=True,
    )
    recos = recos.merge(tmp, how='left', left_on='item_id', right_on='id')
    warnings.filterwarnings("default")
    return recos

def SVD(brons, user, restoran_norm, tmp):
    # Отключить все предупреждения
    warnings.filterwarnings("ignore")
    dataset, ratings = start(brons, user, restoran_norm)
    # Fit model and generate recommendations for all users
    model = PureSVDModel()
    model.fit(dataset)
    recos = model.recommend(
        users=ratings[Columns.User].unique(),
        dataset=dataset,
        k=10,
        filter_viewed=True,
    )
    recos = recos.merge(tmp, how='left', left_on='item_id', right_on='id')
    warnings.filterwarnings("default")
    return recos




