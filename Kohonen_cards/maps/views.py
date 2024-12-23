import matplotlib
matplotlib.use('Agg') 
import time
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from django.shortcuts import render
from django.http import HttpResponse


# Функция для нахождения победителя
def find_winner(point, weights):
    distances = np.linalg.norm(weights - point,axis=1)
    return np.argmin(distances)


# 2. Классическое обучение ("Победитель забирает всё")
def kohonen_training(points, weights, interations=100, alpha=0.5):
    for cur_iteration in range(interations):
        lr = alpha / ((1+cur_iteration) / 2)  # Уменьшаем скорость обучения
        print(lr)
        for point in points:
            winner = find_winner(point, weights)
            weights[winner] += lr * (point - weights[winner])
            #weights[winner] /= np.linalg.norm(weights[winner])  # Нормализация
    return weights


def kohonen_with_fatigue(points, weights, fatigue, interations=100, alpha=0.5):
    for cur_iteration in range(interations):
        lr = alpha / ((1 + cur_iteration) / 2)
        print(lr)
        for point in points:
            distances = np.linalg.norm(weights - point, axis=1) + fatigue
            winner = np.argmin(distances)
            weights[winner] += lr * (point - weights[winner])
            #weights[winner] /= np.linalg.norm(weights[winner])
            fatigue[winner] += 0.5  # Увеличиваем усталость победившего нейрона
    return weights


# Функция для чтения данных из загруженного файла
def parse_txt_file(file):
    points = []
    for line in file:
        try:
            x, y = map(float, line.strip().split())
            points.append([x, y])
        except ValueError:
            continue 
    return np.array(points)


# Представление для загрузки файла и построения графика
def kohonen_view(request):
    if request.method == 'POST' and request.FILES.get('datafile'):
        datafile = request.FILES['datafile']
        points = parse_txt_file(datafile)

        if points.size == 0:
            return HttpResponse("Файл пустой или содержит некорректные данные.", status=400)\

        request.session['points'] = points.tolist()

        # Построение графика
        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], c='blue', label='Объекты')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.legend()
        plt.title("Исходные данные")

        # Сохранение графика в буфер
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        # Кодирование изображения в base64
        plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Возврат графика в ответе
        return render(request, template_name='graph1.html',
                      context={"plot_url": plot_url})

    return render(request, 'upload.html')


def start(request):
    if request.method == 'POST':
        # Считываем количество кластеров из формы
        num_clusters = request.POST.get('num_clusters')

        try:
            num_clusters = int(num_clusters)
        except ValueError:
            num_clusters = None

    points = np.array(request.session.get('points', []))
    weights = np.random.uniform(-1, 1, (num_clusters, 2))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    t1 = time.time()
    classic_weights = kohonen_training(points, weights.copy(),
                                       interations=100, alpha=0.7)
    print("Время обучения классическим алгоритмом: ")
    classic_time = time.time() - t1

    print(weights)

    fatigue = np.zeros(num_clusters)
    t1 = time.time()
    fatigue_weights = kohonen_with_fatigue(points, weights.copy(),
                                           interations=100, alpha=0.1, fatigue=fatigue)
    print("Время обучения алгоритмом с утомляемостью: ")
    fat_time = time.time() - t1

    # Создаем график с классическими весами и весами с утомляемостью
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Объекты')
    plt.scatter(classic_weights[:, 0], classic_weights[:, 1], c='red', label='Классические веса')
    plt.scatter(fatigue_weights[:, 0], fatigue_weights[:, 1], c='yellow', label='Веса с утомляемостью')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Сравнение весов нейронов")

    # Сохранение первого графика в буфер
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plt.close()

    # Кодирование изображения в base64
    plot_url1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')

    # Масштабируем веса для отображения на графиках
    scale_factor = 1
    scaled_weights = fatigue_weights * scale_factor
    scaled_classic = classic_weights * scale_factor

    # Визуализация масштабированных весов с утомляемостью
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Объекты')
    plt.quiver(
        [0] * num_clusters, [0] * num_clusters,
        scaled_weights[:, 0], scaled_weights[:, 1],
        angles='xy', scale_units='xy', scale=1, color='orange', label='Масштабированные веса'
    )
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Масштабированные вектора весов с утомляемостью")

    # Сохранение второго графика в буфер
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    plt.close()

    # Кодирование изображения в base64
    plot_url2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')

    # Визуализация масштабированных весов без утомляемости
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Объекты')
    plt.quiver(
        [0] * num_clusters, [0] * num_clusters,
        scaled_classic[:, 0], scaled_classic[:, 1],
        angles='xy', scale_units='xy', scale=1, color='orange', label='Масштабированные веса'
    )
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Масштабированные вектора весов без утомляемости")

    # Сохранение третьего графика в буфер
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    plt.close()

    # Кодирование изображения в base64
    plot_url3 = base64.b64encode(buffer3.getvalue()).decode('utf-8')

    # Передаем изображения в шаблон
    return render(request, 'result.html', {
        'plot_url1': plot_url1,
        'plot_url2': plot_url2,
        'plot_url3': plot_url3,
        'classic_time': classic_time,
        'fat_time': fat_time
    })
