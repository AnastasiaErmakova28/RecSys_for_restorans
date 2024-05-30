import clear
import discription
import global_top
import similar
import pandas as pd
import ALS
import Item_to_item
import user_to_user
import test
import time
import test_item

start_time = time.time()

#Загрузка данных
raitings_df, people, brons = clear.uploadind_data()

#Очистка данных
test = clear.clear_data(raitings_df)

#Составляем описание
discription = discription.frequency_words(test)
discription.to_excel('discription.xlsx')


#Выводим глобальный топ
top, top_m, restoran_norm, tmp = global_top.avg_rating(raitings_df, brons)
top.to_excel('top_10000.xlsx')
top_m.to_excel('top_m.xlsx')
restoran_norm.to_excel('restoran_norm.xlsx')
tmp.to_excel('tmp.xlsx')

discription = pd.read_excel("discription.xlsx")
discription['text'] = discription['text'].apply(lambda x: eval(x))

top, top_m, restoran_norm, tmp = clear.data()

end_time = time.time()
execution_time = end_time - start_time

print(f"Время загрузки данных: {round(execution_time, 2)} секунд")


#Надем похожие завдеения на основе описания
title = input('Введите название заведение: ')
rec = similar.get_recommendations(title,  top, discription)
rec.to_excel('similar_{}.xlsx'.format(title))


#Рекомендации на основе ALS
id = input("Введите названия заведений через запятую: ").split(',')
rec = ALS.get_recommendations_ALS(top_m, discription, brons, id)
rec.to_excel('recommend_ALS_{}.xlsx'.format(id))



#Рекомендации на основе Item-to-Item
id = input("Введите названия заведений через запятую: ").split(',')
rec = Item_to_item.get_recommendations_item_to_item(top_m, discription, brons, id)
rec.to_excel('recommend_Item_to_Item_{}.xlsx'.format(id))



#Рекомендации на основе user_to_user
id = input("Введите названия заведений через запятую: ").split(',')
rec = user_to_user.get_recommendations_user(top_m, discription, brons, id)
rec.to_excel('recommend_user_to_user_{}.xlsx'.format(id))



#Топ на основе RecTools
user= brons
rec = test.top(brons, user, restoran_norm, tmp)
rec.to_excel('rt_top.xlsx')
print('Готов глобальный топ')

#Рекомендации на основе RecTools
user = brons
rec = test.ALS(brons, user, restoran_norm, tmp)
rec.to_excel('rt_ALS.xlsx')
print('Готовы рекомендации ALS')


#Рекомендации на основе RecTools
user = brons
rec = test.item_to_item(brons, user, restoran_norm, tmp)
rec.to_excel('rt_item_to_item.xlsx')
print('Готовы рекомендации item_to_item')

#Рекомендации на основе RecTools
user = brons
rec = test.SVD(brons, user, restoran_norm, tmp)
rec.to_excel('rt_SVD.xlsx')
print('Готовы рекомендации SVD')

end_time = time.time()
execution_time = end_time - start_time

print(f"Время выполнения: {round(execution_time, 2)} секунд")
