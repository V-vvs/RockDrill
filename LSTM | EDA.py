# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:09:52 2024

@author: vanes
"""
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Arquivos do dataset de treino
file_paths = [
    'data_pdmp1.csv', 'data_pdmp2.csv', 'data_pdmp4.csv', 'data_pdmp5.csv', 'data_pdmp6.csv',
    'data_pin1.csv', 'data_pin2.csv', 'data_pin4.csv', 'data_pin5.csv', 'data_pin6.csv',
    'data_po1.csv', 'data_po2.csv', 'data_po4.csv', 'data_po5.csv', 'data_po6.csv'
]


# Armazenando dados por classe e sensor
class_data_dict = {sensor: {i: [] for i in range(1, 12)} for sensor in ['pdmp', 'pin', 'po']}

# Função para determinar o sensor a partir do nome do arquivo
def get_sensor_from_filename(filename):
    if 'pdmp' in filename:
        return 'pdmp'
    elif 'pin' in filename:
        return 'pin'
    elif 'po' in filename:
        return 'po'
    return None

# Organizando os arquivos e armazenando os dados por classe e sensor
for file_path in file_paths:
    # Ler o arquivo e garantir que todas as linhas tenham o mesmo número de colunas
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Contando o número de colunas esperadas
    expected_columns = len(lines[0].strip().split(','))

    # Ajustando linhas com número incorreto de colunas
    corrected_lines = []
    for line in lines:
        fields = line.strip().split(',')
        if len(fields) != expected_columns:
            # Corrigir ou ignorar a linha
            # Aqui, como exemplo, preenchemos com valores vazios
            fields = fields[:expected_columns] + [''] * (expected_columns - len(fields))
        corrected_lines.append(','.join(fields))

    # Corrigindo os arquivos
    corrected_file_path = file_path.replace('.csv', '_corrigido.csv')
    with open(corrected_file_path, 'w') as f:
        f.write('\n'.join(corrected_lines))

    # Carregando os arquivos corrigidos
    df = pd.read_csv(corrected_file_path, header=None)

    # Gerar nomes das colunas
    num_colunas = df.shape[1]
    colunas = ['label'] + [f'coluna{i}' for i in range(1, num_colunas)]

    # Adicionando os nomes das colunas ao dataframe
    df.columns = colunas

    # Filtrando dados para labels diferentes de 0
    df_filtered = df[df['label'] != 0]

    if df_filtered.empty:
        print(f"Arquivo {file_path} não possui labels diferentes de 0.")
        continue

    # Plotando histogramas da distribuições das classes
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df_filtered)
    plt.title(f'Distribuição das Classes no Dataset {file_path}')
    plt.xlabel('Classe')
    plt.ylabel('Contagem')
    plt.show()

    # Verificando o sensor a partir do nome do arquivo
    sensor = get_sensor_from_filename(file_path)
    if sensor is None:
        print(f"Sensor não identificado no arquivo {file_path}.")
        continue

    # Armazenando os dados por classe e sensor
    classes = df_filtered['label'].unique()
    for cls in classes:
        class_data = df_filtered[df_filtered['label'] == cls]
        if not class_data.empty:
            sample = class_data.iloc[0, 1:].values  # Pegar a primeira amostra da classe
            class_data_dict[sensor][cls].append((file_path, sample))

# Plotando gráficos sobrepostos
def plot_overlapped_graphs(classes_to_plot, sensors=['pdmp', 'pin', 'po']):
    fig, axes = plt.subplots(len(sensors), len(classes_to_plot), figsize=(15, 15), sharex=True, sharey=True)
    for i, sensor in enumerate(sensors):
        for j, cls in enumerate(classes_to_plot):
            ax = axes[i, j]
            data_list = class_data_dict[sensor][cls]
            for file_path, sample in data_list:
                regime = file_path.split(sensor)[1].split('.csv')[0]
                ax.plot(sample, label=f'Regime {regime}')
            if i == 0:
                ax.set_title(f'Classe {cls}')
            if j == 0:
                ax.set_ylabel(sensor.upper())
            if i == len(sensors) - 1:
                ax.set_xlabel('Instante de Tempo')
            ax.legend()
    plt.tight_layout()
    plt.show()

# Plotando sobreposição dos gráficos para cada classe e sensor individualmente
for sensor, class_dict in class_data_dict.items():
    for cls, data_list in class_dict.items():
        if data_list:
            plt.figure(figsize=(12, 6))
            for file_path, sample in data_list:
                regime = file_path.split(sensor)[1].split('.csv')[0]
                plt.plot(sample, label=f'Regime {regime}')
            plt.title(f'Série Temporal para a Classe {cls} - Sensor {sensor.upper()}')
            plt.xlabel('Instante de Tempo')
            plt.ylabel('Valor')
            plt.legend()
            plt.show()

# Plotando gráficos 3x3 para combinações específicas de classes
combinations = [
    [1, 2, 3],
    [1, 4, 5],
    [1, 6, 7],
    [1, 8, 9],
    [1, 10, 11]
]



# Lendo e corrigindo arquivos
def read_and_correct_csv(file_path, max_columns):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Contando o número de colunas esperadas
    expected_columns = len(lines[0].strip().split(','))

    # Ajustando linhas com número incorreto de colunas
    corrected_lines = []
    for line in lines:
        fields = line.strip().split(',')
        if len(fields) != expected_columns:
            fields = fields[:expected_columns] + [''] * (expected_columns - len(fields))
        corrected_lines.append(','.join(fields))

    # Escrevendo o arquivo corrigido temporariamente
    corrected_file_path = file_path.replace('.csv', '_corrigido.csv')
    with open(corrected_file_path, 'w') as f:
        f.write('\n'.join(corrected_lines))

    # Carregando o arquivo corrigido
    df = pd.read_csv(corrected_file_path, header=None)

    # Padronizando o número de colunas para todos
    if df.shape[1] < max_columns:
        df = pd.concat([df, pd.DataFrame(columns=range(df.shape[1], max_columns))], axis=1)

    return df

# verificando o número máximo de colunas
max_columns = 0
for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_columns = len(lines[0].strip().split(','))
    if num_columns > max_columns:
        max_columns = num_columns

# Carregando e preparando os dados
all_data = []
all_labels = []

for file_path in file_paths:
    df = read_and_correct_csv(file_path, max_columns)
    num_colunas = df.shape[1]
    colunas = ['label'] + [f'coluna{i}' for i in range(1, num_colunas)]
    df.columns = colunas

    # Filtrando dados para labels diferentes de 0
    df_filtered = df[df['label'] != 0]
    if df_filtered.empty:
        print(f"Arquivo {file_path} não possui labels diferentes de 0.")
        continue

    # Adicionando os dados e rótulos às listas
    all_data.append(df_filtered.iloc[:, 1:].values)
    all_labels.append(df_filtered['label'].values)

# Convertendo listas em arrays numpy
X = np.vstack(all_data)
y = np.hstack(all_labels)

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste (somente com dados do conjunto de treino, pois são os únicos rotulados)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Redefinindo a forma dos dados para o LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Verificando a forma dos dados
print("Forma de X_train_reshaped:", X_train_reshaped.shape)
print("Forma de X_test_reshaped:", X_test_reshaped.shape)

# Criando o modelo LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Número de classes

# Compilando o modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Avaliando o modelo
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

# Relatório de classificação
print(classification_report(y_test, y_pred_classes))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Novos arquivos não rotulados
new_file_paths = ['data_pdmp3.csv', 'data_pin3.csv', 'data_po3.csv']

# Prevendo novos dados
def predict_new_data(file_paths, model, scaler, max_columns):
    for file_path in file_paths:
        df = read_and_correct_csv(file_path, max_columns)
        num_colunas = df.shape[1]
        colunas = ['coluna{}'.format(i) for i in range(1, num_colunas+1)]
        df.columns = colunas

        X_new = df.values
        X_new_scaled = scaler.transform(X_new)
        X_new_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

        y_pred = model.predict(X_new_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"Previsões para {file_path}:")
        print(y_pred_classes)

predict_new_data(new_file_paths, model, scaler, max_columns)
