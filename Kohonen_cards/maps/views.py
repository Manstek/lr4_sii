import plotly.graph_objects as go
import numpy as np
import time
from django.shortcuts import render
from django.http import HttpResponse

import numpy as np


def find_closest_neuron(input_point, neuron_weights):
    distances = np.linalg.norm(neuron_weights - input_point, axis=1)
    return np.argmin(distances)


# 2. Классическое обучение ("Победитель забирает всё")
def kohonen_train_classic(data_points, neuron_weights, num_iterations, initial_learning_rate=0.5):
    for iteration in range(num_iterations):
        learning_rate = initial_learning_rate / (1 + np.log(1 + iteration))
        for data_point in data_points:
            winning_neuron = find_closest_neuron(data_point, neuron_weights)
            neuron_weights[winning_neuron] += learning_rate * (data_point - neuron_weights[winning_neuron])
    return neuron_weights


def kohonen_train_with_fatigue(data_points, neuron_weights, neuron_fatigue, num_iterations, initial_learning_rate=0.5):
    for iteration in range(num_iterations):
        learning_rate = initial_learning_rate / (1 + np.log(1 + iteration))
        for data_point in data_points:
            adjusted_distances = np.linalg.norm(neuron_weights - data_point, axis=1) + neuron_fatigue
            winning_neuron = np.argmin(adjusted_distances)
            neuron_weights[winning_neuron] += learning_rate * (data_point - neuron_weights[winning_neuron])
            neuron_fatigue[winning_neuron] += 0.5
    return neuron_weights


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
            return HttpResponse("Файл пустой или содержит некорректные данные.", status=400)

        request.session['points'] = points.tolist()

        # Построение графика с помощью plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Объекты', marker=dict(color='blue')))
        fig.update_layout(
            title="Исходные данные",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True
        )

        # Сохранение графика в формате HTML
        graph_html = fig.to_html(full_html=False)

        return render(request, template_name='graph1.html', context={"plot_url": graph_html})

    return render(request, 'upload.html')


def start(request):
    if request.method == 'POST':
        num_clusters = request.POST.get('num_clusters')
        try:
            num_clusters = int(num_clusters)
        except ValueError:
            num_clusters = None

    points = np.array(request.session.get('points', []))
    weights = np.random.uniform(-1, 1, (num_clusters, 2))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)

    # Классическое обучение
    t1 = time.time()
    classic_weights = kohonen_train_classic(points, weights.copy(), num_iterations=100, initial_learning_rate=0.7)
    classic_time = time.time() - t1

    # Обучение с утомляемостью
    fatigue = np.zeros(num_clusters)
    t1 = time.time()
    fatigue_weights = kohonen_train_with_fatigue(points, weights.copy(), num_iterations=100, initial_learning_rate=0.1, neuron_fatigue=fatigue)
    fat_time = time.time() - t1

    # Построение графиков с помощью plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Объекты', marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=classic_weights[:, 0], y=classic_weights[:, 1], mode='markers', name='Классические веса', marker=dict(color='red')))
    fig1.add_trace(go.Scatter(x=fatigue_weights[:, 0], y=fatigue_weights[:, 1], mode='markers', name='Веса с утомляемостью', marker=dict(color='yellow')))
    fig1.update_layout(
        title="Сравнение весов нейронов",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Объекты', marker=dict(color='blue')))
    for i in range(num_clusters):
        fig2.add_trace(go.Scatter(x=[0, fatigue_weights[i, 0]], y=[0, fatigue_weights[i, 1]], mode='lines+markers', name=f'Вектор {i + 1} (утомляемость)'))
    fig2.update_layout(
        title="Масштабированные вектора весов с утомляемостью",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Объекты', marker=dict(color='blue')))
    for i in range(num_clusters):
        fig3.add_trace(go.Scatter(x=[0, classic_weights[i, 0]], y=[0, classic_weights[i, 1]], mode='lines+markers', name=f'Вектор {i + 1} (без утомляемости)'))
    fig3.update_layout(
        title="Масштабированные вектора весов без утомляемости",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    # Передаем графики в шаблон
    return render(request, 'result.html', {
        'plot_url1': fig1.to_html(full_html=False),
        'plot_url2': fig2.to_html(full_html=False),
        'plot_url3': fig3.to_html(full_html=False),
        'classic_time': classic_time,
        'fat_time': fat_time
    })
