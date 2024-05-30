from sklearn.preprocessing import MinMaxScaler



def avg_rating(raitings_df, brons):
    # Расчет среднего рейтинга и количества рейтингов для каждого места
    restoran = raitings_df.groupby(['id', 'adress']).agg({'raiting': ['mean', 'count']}).reset_index()
    restoran.columns = ['id', 'adress', 'avg_rating', 'cnt_rating']
    print(restoran['avg_rating'].hist())

    result = brons.groupby('rest_id').size().reset_index(name='count')

    restoran = restoran.merge(result, how='left', left_on='id', right_on='rest_id')
    restoran['count'] = restoran['count'].fillna(0)

    # Нормализация данных
    scaler = MinMaxScaler()
    restoran_norm = restoran[['id', 'adress', 'avg_rating', 'cnt_rating', 'count']]
    restoran_norm[['avg_rating_norm', 'cnt_rating_norm', 'cnt_brons_norm']] = scaler.fit_transform(
        restoran[['avg_rating', 'cnt_rating', 'count']])

    # Создание нового признака "rank_score" на основе формулы
    restoran_norm['rank_score'] = restoran_norm['avg_rating_norm'] * 0.8 + restoran_norm['cnt_rating_norm'] * 0.15 + \
                                  restoran_norm['cnt_brons_norm'] * 0.05

    # Разделение адреса на город
    restoran_norm['city'] = restoran_norm['adress'].apply(lambda x: x.split(',')[0])

    # Сортировка по rank_score в разрезе по городам
    restoran_norm = restoran_norm.sort_values(by=['city', 'rank_score'], ascending=[True, False])

    restoran_norm = restoran_norm.sort_values(by=['rank_score'], ascending=[False])

    tmp = raitings_df[['id', 'name_ru']]
    tmp = tmp.copy()
    tmp.drop_duplicates(subset=['id'], keep='first', inplace=True)

    # создадим топ 10 000
    top = restoran_norm[['id', 'adress', 'avg_rating', 'cnt_rating', 'count', 'rank_score']]
    top = top.merge(tmp, how='left', on='id')
    top = top.head(10000)

    rest_m = restoran_norm.loc[restoran_norm['city'] == 'Москва']
    top_m = rest_m[['id', 'adress', 'avg_rating', 'cnt_rating', 'count', 'rank_score']]
    top_m = top_m.merge(tmp, how='left', on='id')

    return top, top_m, restoran_norm, tmp
