import pandas as pd

# путь к датасету
data_file_path = 'data.csv'
# прочитать данные в переменную типа DataFrame
houses_data = pd.read_csv(data_file_path)
# размер датасета
print(houses_data.shape)
# печать всех данных
print(houses_data.describe())
# выбираем данные для прогнозирования
y = houses_data.Price
# список столбцов для выборки
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# сама выборка
X = houses_data[features]

from sklearn.tree import DecisionTreeRegressor

# Определяем модель
house_model = DecisionTreeRegressor(random_state=1)

# Обучаем модель
house_model.fit(X, y)

print("Прогноз будет составлен для 5 первых записей:")
print(X.head())
print("Прогноз:")
print(house_model.predict(X.head()))
print("Реальные данные:")
print(y.head())

from sklearn.metrics import mean_absolute_error

predicted_home_prices = house_model.predict(X)
print("MAE1: %d" %mean_absolute_error(y, predicted_home_prices))

from sklearn.model_selection import train_test_split

# разделяем данные на тренироввочный и тестовый набор
# разбиение основано на генераторе случайных чисел
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# определяем модель
house_model = DecisionTreeRegressor(max_leaf_nodes=10, random_state=0)
# обучаем модель
house_model.fit(train_X, train_y)

# получаем среднюю абсолютную ошибку
val_predictions = house_model.predict(val_X)
print("MAE2: %d" %mean_absolute_error(val_y, val_predictions))

# функция для подсчета средней абсолютной ошибки на заданных данных для заданной глубины дерева
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# сравнение MAE для разной глубины дерева
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Глубина дерева: %d  \t\t MAE:  %d" %(max_leaf_nodes, my_mae))

# случайный лес
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
print("MAE RandomForest: %d" %mean_absolute_error(val_y, forest_preds))
