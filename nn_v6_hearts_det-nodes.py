#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:44:24 2023

@author: max
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:08:58 2023

@author: max
"""

"""
Created on Wed Nov  8 19:31:17 2023

Neural Network v4

TODO: speicher Model für spätere Versuche
TODO: Tausche 1) und 2) und vergleiche Visualisierungen
TODO: iris nur mit test.npy durchgeführt. Ist das der ganze Datensatz?

-> KDE vor Imputation: 

Ablauf:
    - Input: Datensatz mit fehlenden Werten => Unsicherheit
    0) lade daten lol
    1) Visualisiere Unsicherheit durch KDE in Inputdaten
    2) Imputiere fehlende Daten mit den begefügten pdf
    3) Trainiere NN mit imputierten Daten
    4) Erstelle MCS
        a) Ziehe aus zuvor erstellten KDE Stichproben
        b) predicte dafür das Ergebnis des Modells
            Änderung zu V2: verwende alle 4 Attribute und überprüfe, ob predictions (GMM) immer noch bei 3 liegt
            -> hat sich geklärt
        c) sichere dieses VOR der Softmax-Funktion (hat sich irgendwie von alleine gelöst)
    5) Erstelle GMM mit den 3 unsicheren Variablen und den gesicherten Ergebnissen 

@author: max
"""
# Heart

import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers
from scipy.stats import gaussian_kde, norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.utils import np_utils
from keras.models import Model
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
%matplotlib qt

SEED = 73
tf.keras.utils.set_random_seed(SEED)

dataset = "heart"
MISSING = 30
path = f'{dataset}/data'

show_model_stats = False

#uncertain_attr = [0, 1, 3]
#uncertain_attr = ['age', 'trestbps', 'chol']
uncertain_attr = [0, 4]
#uncertain_attr = [2, 6]

# Lade Dateien
with open(f"{path}/{dataset}_data_{MISSING}.pkl", 'rb') as file:
    data = pickle.load(file)
    
with open(f"{path}/{dataset}_saved_pdfs_{MISSING}.pkl", 'rb') as file:
    data_pdf = pickle.load(file)
    
    
X = data.iloc[:,:-1]
if dataset == "iris": #TODO: Spalten schon in imputation_eval_main_V2 setzen?
    X.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
y = data.iloc[:,-1]
#income_mapping = {'<=50K': 1, '>50K': 0}
#y = y.apply(lambda x: income_mapping[x])
    
plt_rows = int(np.ceil(np.sqrt(len(data.columns))))
plt_cols = int(np.ceil(len(data.columns)/plt_rows))

kdes = []

# KDE's
fig, axes = plt.subplots(nrows=plt_rows, ncols=plt_cols, figsize=(12, 8))
for i, (column, column_data) in enumerate(X.items()):
    column_data = X[column].dropna()

    # Erstellen der KDE für die aktuelle Spalte
    if column_data.dtype == "object": #sollte durch aufteilen in X,Y nicht mehr vorkommen
        continue
    kde = gaussian_kde(column_data)
    kdes.append(kde)

    # Erstellen eines Arrays von Werten, um die KDE zu bewerten
    #x = np.linspace(column_data.min(), column_data.max(), 1000)

    # Berechnen der PDF-Werte für die KDE
    #pdf_values = kde(x)

    # Erstellen eines Plots für die KDE
    row = i // plt_cols
    col = i % plt_cols
    ax = axes[row, col]
    #ax.plot(x, pdf_values, label=f'KDE für {column}')
    '''
    NOTE: seaborn verwendet ebenfalls gaussian_kde mit @data und die Visualisierung
    mit seaborn sollte daher genau der berechneten kde entsprehcne
    '''
    mean = round(kde.resample().mean(), 4)
    std = round(kde.resample().std(), 4)
    kurt = round(column_data.kurt(), 4)
    skew = round(column_data.skew(), 4)
    print(f"{column}: {mean} (mean), {std} (std), {kurt} (kurt), {skew} (skew)")
    sns.set_palette("plasma", n_colors=1) #funktionuert erst beim 2ten mal?
    sns.kdeplot(data=column_data, ax=ax, label=f'KDE für {column}', fill=True)
    ax.set_title(f'KDE für {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Dichte')

# Entfernen von leeren Subplots
for i in range(len(X.columns), plt_rows * plt_cols):
    fig.delaxes(axes.flatten()[i])

# Anpassen der Layoutparameter
plt.tight_layout()
plt.show()

X.hist()
plt.show()

# Daten vorbereiten
def impute(data, pdf_dict):
    for row_index, row in data.iterrows():
        missing_indices = row.index[row.isna()]
        for col_index in missing_indices:
            pdf = pdf_dict.get(row_index)[0]
            resample = pdf.resample(size=1)
            data.at[row_index, col_index] = resample[0]

impute(X, data_pdf)

# data_australian.to_csv("datasets/data_australian_imputed.csv")

start_time = time.time()
# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

if dataset == "housing" or dataset == "heart_":
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    sc_y = MinMaxScaler()
    y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sc_y.transform(y_test.values.reshape(-1, 1))

if dataset == "heart_":
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sc_y.transform(y_test.values.reshape(-1, 1))

#TODO: Wofür war das nochmal?
if dataset == "iris":
    y_train=np_utils.to_categorical(y_train, num_classes=3)
    y_test=np_utils.to_categorical(y_test, num_classes=3)

# Keras model
model = tf.keras.models.Sequential([
    layers.Input(X_train.shape[1:]),
    layers.Dense(32, activation='relu'),
    layers.Dense(8, activation='relu'),
    #layers.Dense(1) # regression
    #layers.Dense(3, activation='softmax')
    layers.Dense(1, activation='sigmoid') #heart
])

#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # regression
model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy']) #heart
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), verbose=False)

#model.save('saved_model/model_iris-313c.keras')

if show_model_stats:
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
print("accuracy:", history.history['accuracy'][-1])

end_time = time.time()
model_creation_time = end_time - start_time

print(f'Model created in {model_creation_time}s')

# Analyse der Schichten
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Anzahl der Monte Carlo-Durchläufe
num_sim = 2000
sim_data = []
sim_results = [] # sollte num_sim Einträge mit 3 Werten für die Klassen enthalten (Iris)

for i in range(num_sim):
    sim = []
    for kde in kdes:
        sample = kde.resample(1)[0]
        sim.append(sample[0])
    sim_data.append(sim)
        
pred_probs = model.predict(sim_data, verbose=False)
sim_data = np.array(sim_data)

# Aktivierungen für jedes Layer
activations = activation_model.predict(sim_data)
activation_before_sm = None

for i, activation in enumerate(activations):
    if i == (len(activations)-2):
        if dataset == "heart":
            activation_before_sm = activation

            fig, ax = plt.subplots(figsize=(10, 6))
            #ax.hist(activation.flatten(), bins=100, density=True, alpha=0.7, label="Histogramm")
            sns.kdeplot(data=activation.flatten(), fill=True, ax=ax)
            ax.set_title("KDE vor SoftMax")
            ax.set_xlabel('Vorhersage des vorletzten Layer')
            ax.set_ylabel('Dichte')
            plt.show()
        
    print(f"Schicht {i}: {activation.shape}")

# Histogram
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(pred_probs.flatten(), bins=100, density=True, alpha=0.7, label="Histogramm")
sns.kdeplot(data=pred_probs.flatten(), fill=True, ax=ax, label="KDE")
ax.set_title("Histogramm/KDE nach SoftMax")
ax.set_xlabel('Modellvorhersage')
ax.set_ylabel('Dichte')
plt.legend()
plt.show()

def calculate_bic(data, component_range):
    bic_values = []
    
    for n_components in component_range:
        gmm = GaussianMixture(n_components=n_components, random_state=SEED)
        gmm.fit(data)
        bic_values.append(gmm.bic(data))

    return bic_values

component_range = range(1, 11)
bics = calculate_bic(activation_before_sm, component_range)
fig = plt.figure()
plt.plot(component_range, bics)
plt.xlabel("Anzahl an Komponenten")
plt.ylabel("BIC")
plt.show()

n_components = component_range[bics.index(min(bics))]
n_components = 1 #hearts
print(f"using {n_components} components...")

pred_probs = pred_probs.reshape(-1, 1)
# Visualisierung als Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_components, warm_start=True, random_state=SEED)
gmm.fit(activation_before_sm)

bic = gmm.bic(activation_before_sm)
aic = gmm.aic(activation_before_sm)

print("Means:", gmm.means_)
print("Spur", np.trace(gmm.covariances_))
print(f"{bic} (bic), {aic} (aic)")


# --------------------------------------
# GMM (Anzahl der Attribute irrelevant)
# --------------------------------------
plt.figure()


#x = np.linspace(-100, 600, 1000)
x = np.linspace(0, 400, 500)
gmm_curve = np.zeros_like(x)

for i in range(n_components):
    mean = gmm.means_[i]
    std = np.sqrt(np.diag(gmm.covariances_[i]))
    weight = gmm.weights_[i] #?
    
    # Erzeugen von x-Werten für die Gesamtkurve
    #x = np.linspace(X.iloc[i].min(), X.iloc[i].max(), 100) #es wird nicht der ganze Bereich dargestellt
    
    # Initialisieren einer leeren Kurve
    total_curve = np.zeros_like(x)
    
    #WORKING
    '''
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)  # Erzeugen von x-Werten für die PDF
    
    # Berechnen der PDF für die Komponente
    y = norm.pdf(x, mean, std) * weight
    
    plt.plot(x, y, label=f'Component {i+1}')  # Plot der PDF für die Komponente
    '''
    
    for j in range(activation_before_sm.shape[1]):
        #x = np.linspace(X.iloc[j].min(), X.iloc[j].max(), 100) #es wird nicht der ganze Bereich dargestellt
        
        # Berechnen der PDF für die Komponente
        y = norm.pdf(x, mean[j], np.sqrt(gmm.covariances_[i][j, j])) * weight
        total_curve += y
        
        '''!!! Interessant'''
        #plt.plot(x, y, label=f'Komponente {i+1}, Feature {j+1}')  #Zeigt einezlne Features
    
    plt.plot(x, total_curve, label=f'Komponente {i+1}', color=sns.color_palette("plasma", n_components)[i])
    plt.fill_between(x, total_curve, alpha=0.3, color=sns.color_palette("plasma", n_components)[i])
    #sns.kdeplot(x=x, data=total_curve, label=f'Komponente {i+1}') # funktioniert nicht ganz
    gmm_curve += total_curve

#gmm_color = sns.color_palette("plasma", n_components)[-1]
gmm_color = "red"
plt.plot(x, gmm_curve, label=f'GMM', color=gmm_color)
plt.fill_between(x, gmm_curve, alpha=0.3, color=gmm_color)
#sns.kdeplot(x=x, data=gmm_curve, label=f'GMM') #funktioniert irgendwie nicht ganz
plt.legend()
plt.xlabel('Datenbereich')
plt.ylabel('PDF')
plt.title('Wahrscheinlichkeitsdichtefunktionen für GMM-Komponenten')
plt.show()

gmm_predictions = gmm.predict(activation_before_sm)

plt.figure(figsize=(10, 6))
sns.histplot(activation_before_sm.flatten(), bins=100, kde=True, label="Vorhersagen")

#for i in range(n_components):
    #plt.axvline(gmm.means_[i], color=sns.color_palette("plasma", n_components)[i], linestyle="dashed", linewidth=2, label=f"GMM Komponente {i+1}")
    
plt.title("Vorhersagen und GMM-Komponenten")
plt.xlabel('Vorhersagen')
plt.ylabel('Häufigkeit')
plt.legend()
plt.show()

gmm_models = []

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
for dim in range(activation_before_sm.shape[1]):
    
    gmm = GaussianMixture(n_components=3)
    gmm.fit(activation_before_sm[:, dim].reshape(-1,1))
    gmm_models.append(gmm)
    if dim not in [0, 2, 3]:
        continue
    #plt.hist(activation_before_sm[:, dim], bins=100, density=True, alpha=0.7, label=f"Histogramm {dim+1}")
    x = np.linspace(-10, 70, 100)
    #y = np.exp(gmm_models[dim].score_samples(x.reshape(-1, 1)))
    y = np.exp(gmm_models[dim].score_samples(x.reshape(-1, 1)))
    ax.plot(x, y, label=f'GMM Fit Dim {dim+1}', color=sns.color_palette("plasma", 4)[dim])
    ax.fill_between(x, y, alpha=0.3, color=sns.color_palette("plasma", 4)[dim])
    
    #plt.plot(x, y, label="GMM Fit")
    #plt.title(f"GMM Git for Output Dimension {dim+1}")    
    #plt.legend()
    #plt.show()
    
ax.set_xlabel('Ausgabe Dimension')
ax.set_ylabel('Dichte')
ax.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(activation_before_sm[:, 0], activation_before_sm[:, 2], activation_before_sm[:, 3])
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 3')
ax.set_zlabel('Neuron 4')
plt.show()

# --------------------------------------
# GMM ENDE
# --------------------------------------

# --------------------------------------
# 3D Scatter (für 3 Attribute)
# --------------------------------------
if len(uncertain_attr) == 3:
    x_vals = X.iloc[:, uncertain_attr[0]].values
    y_vals = X.iloc[:, uncertain_attr[1]].values
    z_vals = X.iloc[:, uncertain_attr[2]].values
    
    # Vorhersage der Zuordnung zu den Komponenten für jeden Datenpunkt
    labels = gmm.predict(X.iloc[:, uncertain_attr])
    
    # Festlegen der Farbpalette auf "plasma"
    sns.set_palette("plasma", n_colors=len(np.unique(labels)))
    
    # Plotten der Ergebnisse als 3D-Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3D-Achsen hinzufügen
    
    for i in np.unique(labels):
        indices = labels == i
        ax.scatter(x_vals[indices], y_vals[indices], z_vals[indices], label=f'Komponente {i+1}')
    
    ax.set_xlabel(X.columns[uncertain_attr[0]])
    ax.set_xticks([30, 70])
    ax.set_ylabel(X.columns[uncertain_attr[1]])
    ax.set_zlabel(X.columns[uncertain_attr[2]])
    plt.legend(loc='best')
    plt.title('3D-Streuplot für die GMM Komponenten des Iris-Datensatz')
    plt.show()
    
# --------------------------------------
# 3D Scatter ENDE
# --------------------------------------

# --------------------------------------
# 3D Surface (für 2 Attribute, weil 3 irgendwie nicht gehen... ¯\(°_o)/¯ )
# --------------------------------------

if len(uncertain_attr) == 2 and False:
    
    # Erstelle Beispieldaten
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    x, y = np.meshgrid(x, y)
    
    '''
    # Erstelle 3D-Plot: Probabilities
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    proba = gmm.predict_proba(np.column_stack((x.ravel(), y.ravel())))
    for i in range(2):
        z = proba[:, i]
        z = z.reshape(x.shape)
        
        ax.plot_surface(x, y, z, cmap='plasma', alpha=0.7, label=f'Komponente {i+1}')
    
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_zlabel('Wahrscheinlichkeit')
    plt.title('3D-Oberflächenplot der Wahrscheinlichkeiten für GMM-Komponenten')
    plt.show()
    '''
    
    # Erstelle 3D-Plot: Log-Likelihood
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Berechne die Log-Likelihood für jedes Gitterpunkt
    z = gmm.score_samples(np.column_stack((x.ravel())))
    z = z.reshape(x.shape)
        
    ax.plot_surface(x, y, z, cmap='plasma', alpha=0.7, label=f'Komponente {i+1}')
    
    ax.set_xlabel(X.columns[uncertain_attr[0]])
    ax.set_ylabel(X.columns[uncertain_attr[1]])
    ax.set_zlabel('Log-Likelihood')
    plt.title('3D-Oberflächenplot der Log-Likelihood für GMM-Komponenten')
    plt.show()
    
    
    # Erstelle Beispieldaten
    x = np.linspace(0, 100, 1000)
    y = np.linspace(0, 600, 1000)
    x, y = np.meshgrid(x, y)
    # Erstelle 3D-Plot: PDF
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Berechne die Wahrscheinlichkeitsdichte für jedes Gitterpunkt
    pdf_values = np.exp(gmm.score_samples(np.column_stack((x.ravel(), y.ravel()))))
    pdf_values = pdf_values.reshape(x.shape)
    
    # Plot der Wahrscheinlichkeitsdichte
    ax.plot_surface(x, y, pdf_values, cmap='plasma', alpha=0.7, label='PDF')
    
    ax.set_xlabel(X.columns[uncertain_attr[0]])
    ax.set_ylabel(X.columns[uncertain_attr[1]])
    ax.set_zlabel('Wahrscheinlichkeitsdichte')
    plt.title('3D-Oberfächenplot der Wahrscheinlichkeitsdichte für GMM-Komponenten')
    plt.show()

# --------------------------------------
# 3D Surface ENDE
# --------------------------------------
