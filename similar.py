from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def similar_restaurant(top, discription):
    similar_df = top.merge(discription, how='left', on='id')
    similar_df['text_str'] = similar_df['text'].apply(lambda x: ' '.join([y.strip().replace(' ', '') for y in x]))
    similar_df.to_excel('similar.xlsx')

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(similar_df['text_str'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(similar_df.index, index=similar_df['id']).drop_duplicates()

    cosine_sim_df = pd.DataFrame(cosine_sim)
    cosine_sim_df.columns = indices.index

    cosine_sim_df['title'] = indices.index
    cosine_sim_df = cosine_sim_df.set_index('title')

    return cosine_sim_df, similar_df

def get_recommendations(rest, top, discription):
    cosine_sim_df, similar_df = similar_restaurant(top, discription)
    title = similar_df[similar_df['name_ru'] == rest].id.iloc[0]

    indices = pd.Series(similar_df.index, index=similar_df['id']).drop_duplicates()
    idx = indices[title]

    # Получаем похожести для этого места
    sim_scores = list(enumerate(cosine_sim_df.loc[title]))

    # Сортируем фильмы, основываясь на похожести
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    # Получаем индексы Ресторана
    rest_indices = [i[0] for i in sim_scores]


    return similar_df.iloc[rest_indices]