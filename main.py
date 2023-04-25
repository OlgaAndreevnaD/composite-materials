from tensorflow import keras
import pandas as pd
import skops.io as sio
import warnings
warnings.filterwarnings("ignore")


def ask_data():
    part_data = {}
    columns = ['Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки',
               'Плотность, кг/м3', 'модуль упругости, ГПа', 'Количество отвердителя, м.%',
               'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
               'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
               'Прочность при растяжении, МПа', 'Потребление смолы, г/м2']

    for column in columns:
        x = float(input(f'{column}: '))
        part_data.update({column: x})
    data = pd.DataFrame([part_data], columns=columns)

    return data


if __name__ == '__main__':
    data = ask_data()

    model = keras.models.load_model('D:/composite materials/best_model')
    scaler = sio.load('D:/composite materials/std_scaler.bin', trusted=True)

    data = scaler.transform(data.to_numpy())

    prediction = model.predict(data).flatten()

    print(f"Соотношение матрица-наполнитель: {prediction.item():.5f}")
