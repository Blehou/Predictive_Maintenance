#!/usr/bin/env python
# coding: utf-8
# @author: konain

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


# Partie 1

## Data Visualisation

def load_data(_type, file):
    """
    Load data based on the type: training or test dataset.
    """
    if _type == "training":
        df = pd.read_csv(file)
        _id, cycle = df["id"], df["cycle"]
        sensor1, sensor2, sensor3, sensor4 = df["s1"], df["s2"], df["s3"], df["s4"]
        ttf, label = df["ttf"], df["label_bnc"]
        return _id, cycle, sensor1, sensor2, sensor3, sensor4, ttf, label
        
    elif _type == "test":
        df = pd.read_csv(file)
        __id, cycle = df["id"], df["cycle"]
        sensor1, sensor2, sensor3, sensor4 = df["s1"], df["s2"], df["s3"], df["s4"]
        return _id, cycle, sensor1, sensor2, sensor3, sensor4


def plot_dataset(file, _engineID):
    """
    Plot sensor data for a specific engine ID from the training dataset.
    """
    # Charger les données
    _id, cycle, sensor1, sensor2, sensor3, sensor4, ttf, label = load_data("training", file)
    engine_data = _id == _engineID  # Filtrer les données pour l'ID moteur spécifié
    bncInds = label.astype(bool)    # Indices des moteurs défaillants

    # Créer une figure avec 4 sous-graphes (2x2)
    plt.figure(figsize=(10, 8))

    # Sensor 1
    plt.subplot(2, 2, 1)
    plt.plot(cycle[engine_data & ~bncInds], sensor1[engine_data & ~bncInds], 'b-', label='S1 good data')
    plt.plot(cycle[engine_data & bncInds], sensor1[engine_data & bncInds], 'r-', label='S1 faulty engine')
    plt.title('Sensor 1')
    plt.ylabel('Sensor values')
    plt.xlabel('Cycles')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Sensor 2
    plt.subplot(2, 2, 2)
    plt.plot(cycle[engine_data & ~bncInds], sensor2[engine_data & ~bncInds], 'b-', label='S2 good data')
    plt.plot(cycle[engine_data & bncInds], sensor2[engine_data & bncInds], 'r-', label='S2 faulty engine')
    plt.title('Sensor 2')
    plt.ylabel('Sensor values')
    plt.xlabel('Cycles')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Sensor 3
    plt.subplot(2, 2, 3)
    plt.plot(cycle[engine_data & ~bncInds], sensor3[engine_data & ~bncInds], 'b-', label='S3 good data')
    plt.plot(cycle[engine_data & bncInds], sensor3[engine_data & bncInds], 'r-', label='S3 faulty engine')
    plt.title('Sensor 3')
    plt.ylabel('Sensor values')
    plt.xlabel('Cycles')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Sensor 4
    plt.subplot(2, 2, 4)
    plt.plot(cycle[engine_data & ~bncInds], sensor4[engine_data & ~bncInds], 'b-', label='S4 good data')
    plt.plot(cycle[engine_data & bncInds], sensor4[engine_data & bncInds], 'r-', label='S4 faulty engine')
    plt.title('Sensor 4')
    plt.ylabel('Sensor values')
    plt.xlabel('Cycles')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Améliorer l'affichage avec tight_layout pour éviter le chevauchement
    plt.suptitle(f'Sensor Data for Engine {_engineID}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste pour ne pas couvrir le titre principal
    plt.show()


plot_dataset(file="train_selected.csv", _engineID=1)


def plot_advanced_dataset(file, _engineID, _option="TTF_vs_Cycle"):
    """
    Advanced visualizations for a specific engine ID from the training dataset.
    """
    # Charger les données
    _id, cycle, sensor1, sensor2, sensor3, sensor4, ttf, label = load_data("training", file)
    engine_data = _id == _engineID  # Filtrer les données pour l'ID moteur spécifié
    
    # Filtrer les points où le moteur est défaillant (label == 1) et où il est fonctionnel (label == 0)
    faulty = engine_data & (label == 1)
    normal = engine_data & (label == 0)

    if _option == "TTF_vs_Cycle":
        # 1. Time-to-Failure (TTF) vs Cycles avec deux couleurs et scatter plot
        plt.figure(figsize=(6, 5))
        
        # Scatter plot pour les moteurs fonctionnels (label == 0) en vert
        plt.scatter(cycle[normal], ttf[normal], color='green', label='Normal (label=0)', alpha=0.6)
        
        # Scatter plot pour les moteurs défaillants (label == 1) en rouge
        plt.scatter(cycle[faulty], ttf[faulty], color='red', label='Faulty (label=1)', alpha=0.6)
        
        plt.title(f'Time-to-Failure (TTF) vs. Cycles for engine {_engineID}')
        plt.xlabel('Cycles')
        plt.ylabel('Time-to-Failure (TTF)')
        plt.grid(True)
        plt.legend()
        plt.show()

    elif _option == "Sensor_vs_TTF":
        # 2. Comportement TTF (Time-to-Failure) en fonction des capteurs 
        plt.figure(figsize=(8, 6))
        
        plt.subplot(2, 2, 1)
        plt.scatter(sensor1[faulty], ttf[faulty], color='red', label='Faulty (label=1)', alpha=0.6)
        plt.scatter(sensor1[normal], ttf[normal], color='blue', label='Normal (label=0)', alpha=0.6)
        plt.title(f'Time-to-Failure (TTF) vs. Sensor 1 for engine {_engineID}')
        plt.xlabel('Sensor 1')
        plt.ylabel('Time-to-Failure (TTF)')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(sensor2[faulty], ttf[faulty], color='red', label='Faulty (label=1)', alpha=0.6)
        plt.scatter(sensor2[normal], ttf[normal], color='blue', label='Normal (label=0)', alpha=0.6)
        plt.title(f'Time-to-Failure (TTF) vs. Sensor 2 for engine {_engineID}')
        plt.xlabel('Sensor 2')
        plt.ylabel('Time-to-Failure (TTF)')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.scatter(sensor3[faulty], ttf[faulty], color='red', label='Faulty (label=1)', alpha=0.6)
        plt.scatter(sensor3[normal], ttf[normal], color='blue', label='Normal (label=0)', alpha=0.6)
        plt.title(f'Time-to-Failure (TTF) vs. Sensor 3 for engine {_engineID}')
        plt.xlabel('Sensor 3')
        plt.ylabel('Time-to-Failure (TTF)')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.scatter(sensor4[faulty], ttf[faulty], color='red', label='Faulty (label=1)', alpha=0.6)
        plt.scatter(sensor4[normal], ttf[normal], color='blue', label='Normal (label=0)', alpha=0.6)
        plt.title(f'Time-to-Failure (TTF) vs. Sensor 4 for engine {_engineID}')
        plt.xlabel('Sensor 4')
        plt.ylabel('Time-to-Failure (TTF)')
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()

plot_advanced_dataset(file="train_selected.csv",_engineID=3, _option="Sensor_vs_TTF")

plot_advanced_dataset(file="train_selected.csv",_engineID=1)

## Corrélation entre les données

# Chargement des données
df = pd.read_csv("train_selected.csv")

# Calcul des corrélations
corr_matrix = df[['s1', 's2', 's3', 's4', 'ttf']].corr()
print(corr_matrix)

# Customisation de la heatmap
plt.figure(figsize=(8, 6))

# Palette et annotations ajustées
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, annot_kws={"size": 12}, 
            linewidths=0.5, linecolor='black', square=True, cbar_kws={"shrink": 0.75})

# Titres et ajustement des axes
plt.title("Correlation matrix: Sensors and TTF", fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.show()


# Model Selection

# Lecture ou chargement des données
data_train = pd.read_csv("train_selected.csv")
data_test = pd.read_csv("test_selected.csv")

# Récupération des features (capteurs) et/ou la cible (TTF)
# Donnée d'entrainement
sensors = data_train[['s1', 's2', 's3', 's4']].to_numpy()
ttf = data_train['ttf'].to_numpy()

# Donnée de test
X_test_1 = data_test[['s1', 's2', 's3', 's4']].to_numpy()

ttf_test = []

with open("PM_truth.txt") as file:
    content = file.read()
    # diviser la chaine en sous chaine 
    content = content.split(" \n")
    
    # supprimer le dernier element
    content.pop()
    # récupérer le TTF
    for element in content:
        ttf_test.append(int(element))
        
    file.close()

### Séparation des données (train_test_plit)

# Répartition en donnée d'entrainement et de validation
X_train, X_val, y_train, y_val = train_test_split(sensors, ttf, test_size=0.2, random_state=42)

print(f"Training data size = {X_train.shape}")
print(f"Training data size = {X_train.shape}")
print(f"Validation data size = {X_val.shape}")
print(f"Validation data size = {X_val.shape}")

# Calcul du nombre d'échantillons
train_size = X_train.shape[0]
val_size = X_val.shape[0]
total_size = train_size + val_size

# Afficher la taille des données d'entraînement et validation en pourcentage
train_percent = (train_size / total_size) * 100
val_percent = (val_size / total_size) * 100

# Affichage avec un Pie Chart pour visualiser la répartition
plt.figure(figsize=(8, 6))
labels = ['Training set', 'Validation set']
sizes = [train_percent, val_percent]
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)  # Pour accentuer la partie "Training set"

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title("Split of samples between training and validation", fontsize=12)
plt.show()

### Normalisation des données

# Nomarlisation des données
## StandardScaler 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_1)


## Support Vector Regressor (SVR)

# Initialiser le modèle SVR avec des hyperparamètres manuels
svr_model = SVR(cache_size=1000)
    
# Entraîner le modèle sur les données d'entraînement
svr_model.fit(X_train_scaled, y_train)

# Evaluer le modèle
score_svr = svr_model.score(X_val_scaled, y_val)

# Prediction
y_predict = svr_model.predict(X_val_scaled)

print(f"Sensors (s1, s2, s3, s4) : {X_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_predict[0]}")
print(f"Ground truth TTF : {y_val[0]} \n")

print(f"SVR score : {score_svr} \n")

mae = mean_absolute_error(y_val, y_predict)
rmse = np.sqrt(mean_squared_error(y_val, y_predict))
r2 = r2_score(y_val, y_predict)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

### GridSearchCV : SVR

# Définir la grille d'hyperparamètres à tester
param_grid_svr = {
    'C': [1.0, 2.0, 10.0],  
    'gamma': [1, 2, 5, 'scale', 'auto'], 
    'epsilon': [0.1, 0.2, 0.5, 1.0]  # Ajout du paramètre epsilon avec des valeurs
}

# Initialiser le GridSearchCV avec le modèle SVR et la grille d'hyperparamètres
grid_search_svr = GridSearchCV(estimator=svr_model, param_grid=param_grid_svr, cv=5, scoring='r2', n_jobs=-1, verbose=2)

# Lancer la recherche des meilleurs hyperparamètres sur les données d'entraînement
grid_search_svr.fit(X_train_scaled, y_train)

# Afficher les meilleurs hyperparamètres et le meilleur score
print(f"Best parameters : {grid_search_svr.best_params_}")
print(f"Best score : {grid_search_svr.best_score_}")

### Best estimator : SVR
print(grid_search_svr.best_estimator_)

# Utiliser le meilleur modèle pour faire des prédictions sur les données de validation
best_svr_model = grid_search_svr.best_estimator_
y_pred_val_svr = best_svr_model.predict(X_val_scaled)

# Évaluer les performances sur les données de validation
score_val = best_svr_model.score(X_val_scaled, y_val)
print(f"Score on validation data : {score_val}\n")

print(f"Sensors (s1, s2, s3, s4) : {X_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_pred_val_svr[0]}")
print(f"Ground truth TTF : {y_val[0]} \n")

mae = mean_absolute_error(y_val, y_pred_val_svr)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_svr))
r2 = r2_score(y_val, y_pred_val_svr)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

### Fine Tuning

# Initialiser le modèle SVR avec des hyperparamètres manuels
svr_model = SVR(C=100.0, cache_size=1000, epsilon=1.0, gamma="scale")

# Entraîner le modèle sur les données d'entraînement
svr_model.fit(X_train_scaled, y_train)

# Evaluer le modèle
score = svr_model.score(X_val_scaled, y_val)

# Prediction
y_prediction_tun = svr_model.predict(X_val_scaled)

print(f"Sensors (s1, s2, s3, s4) : {X_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_prediction_tun[0]}")
print(f"Ground truth TTF : {y_val[0]} \n")

print(f"SVR score : {score} \n")

mae = mean_absolute_error(y_val, y_prediction_tun)
rmse = np.sqrt(mean_squared_error(y_val, y_prediction_tun))
r2 = r2_score(y_val, y_prediction_tun)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

### Cross-validation 

# Utilisation de RMSE comme fonction de coût
mse_scores = cross_val_score(estimator=svr_model, X=X_train_scaled, y=y_train, cv=5, 
                             scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Conversion des scores MSE en RMSE
rmse_scores = np.sqrt(-mse_scores)  # Prendre la racine carrée de la valeur négative

# Utilisation de MAE comme fonction de coût
mae_scores = cross_val_score(estimator=svr_model, X=X_train_scaled, y=y_train, cv=5, 
                             scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)

mae_scores = -mae_scores

# Affichage des scores RMSE pour chaque fold
print(f"RMSE pour chaque fold : {rmse_scores}")
print(f"RMSE moyen : {rmse_scores.mean()}\n")

# Affichage des scores MAE pour chaque fold
print(f"MAE pour chaque fold : {mae_scores}")
print(f"MAE moyen : {mae_scores.mean()}")


### Random Forest Regressor

# Initialiser le modèle RandomForest avec des hyperparamètres
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=16, n_jobs=-1, verbose=1)

# Entraîner le modèle sur les données d'entraînement
rf_model.fit(X_train_scaled, y_train)

# Evaluer le modèle
score_rf = rf_model.score(X_val_scaled, y_val)

# Prediction
y_prediction_rf = rf_model.predict(X_val_scaled)

print(f"Sensors (s1, s2, s3, s4) : {X_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_prediction_rf[0]}")
print(f"Ground truth TTF : {y_val[0]} \n")
print(f"Random Forest score : {score_rf} \n")

mae = mean_absolute_error(y_val, y_prediction_rf)
rmse = np.sqrt(mean_squared_error(y_val, y_prediction_rf))
r2 = r2_score(y_val, y_prediction_rf)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

### GridSearchCV : Random Forest Regressor

# Définir la grille d'hyperparamètres à tester pour Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 500],  # Nombre d'arbres
    'criterion': ['squared_error', 'absolute_error'],  # Fonction de perte
    'min_samples_split': [2, 5, 10],  # Minimum d'échantillons pour diviser un nœud
    'min_impurity_decrease': [0.0, 0.01, 0.1]  # Réduction minimale d'impureté
}

# Initialiser le GridSearchCV avec le modèle Random Forest Regressor et la grille d'hyperparamètres
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='r2', n_jobs=-1, verbose=2)

# Lancer la recherche des meilleurs hyperparamètres sur les données d'entraînement
grid_search_rf.fit(X_train_scaled, y_train)

# Afficher les meilleurs hyperparamètres et le meilleur score
print(f"Best parameters : {grid_search_rf.best_params_}")
print(f"Best score : {grid_search_rf.best_score_}")


### Best estimator : Random Forest Regressor
print(grid_search_rf.best_estimator_)

# Utiliser le meilleur modèle pour faire des prédictions sur les données de validation
best_rf_model = grid_search_rf.best_estimator_  # Récupérer le meilleur modèle
y_pred_val_rf = best_rf_model.predict(X_val_scaled)

# Évaluer les performances sur les données de validation

rf_score_val = best_rf_model.score(X_val_scaled, y_val)
print(f"Score on validation data : {rf_score_val}\n")

print(f"Sensors (s1, s2, s3, s4) : {X_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_pred_val_rf[0]}")
print(f"Ground truth TTF : {y_val[0]} \n")

mae = mean_absolute_error(y_val, y_pred_val_rf)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_rf))
r2 = r2_score(y_val, y_pred_val_rf)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

### Cross validation

# Utilisation de RMSE comme fonction de coût
mse_scores = cross_val_score(estimator=rf_model, X=X_train_scaled, y=y_train, cv=5, 
                             scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Conversion des scores MSE en RMSE
rmse_scores = np.sqrt(-mse_scores)  # Prendre la racine carrée de la valeur négative

# Utilisation de MAE comme fonction de coût
mae_scores = cross_val_score(estimator=rf_model, X=X_train_scaled, y=y_train, cv=5, 
                             scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)

mae_scores = -mae_scores

# Affichage des scores RMSE pour chaque fold
print(f"RMSE pour chaque fold : {rmse_scores}")
print(f"RMSE moyen : {rmse_scores.mean()}\n")

# Affichage des scores MAE pour chaque fold
print(f"MAE pour chaque fold : {mae_scores}")
print(f"MAE moyen : {mae_scores.mean()}")


## Prédiction avec les données de test

### SVR
ttf_pred_svr = svr_model.predict(X_test_scaled)

mae = mean_absolute_error(ttf_test, ttf_pred_svr)
rmse = np.sqrt(mean_squared_error(ttf_test, ttf_pred_svr))
r2 = r2_score(ttf_test, ttf_pred_svr)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

# Comparer les prédictions des modèles avec les valeurs réelles
plt.figure(figsize=(8, 6))
plt.plot(ttf_pred_svr, label="Predicted TTF", color='tab:blue', marker='o', markersize=5)
plt.plot(ttf_test, label="Ground truth TTF", color='tab:red', marker='o',markersize=4)
plt.title("Comparison of predictions with ground truth (TTF)")
plt.xlabel("Index")
plt.ylabel("Time to Failure (TTF)")
plt.legend()
plt.show()

### Random Forest Regressor
ttf_pred_ = rf_model.predict(X_test_scaled)

mae = mean_absolute_error(ttf_test, ttf_pred_)
rmse = np.sqrt(mean_squared_error(ttf_test, ttf_pred_))
r2 = r2_score(ttf_test, ttf_pred_)

print(f"MAE : {mae}")
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

# Comparer les prédictions des modèles avec les valeurs réelles
plt.figure(figsize=(8, 6))
plt.plot(ttf_pred_, label="Predicted TTF", color='tab:blue', marker='o', markersize=5)
plt.plot(ttf_test, label="Ground truth TTF", color='tab:red', marker='o',markersize=4)
plt.title("Comparison of predictions with ground truth (TTF)")
plt.xlabel("Index")
plt.ylabel("Time to Failure (TTF)")
plt.legend()
plt.show()


# Partie 2

## Data visualisation
def plot_advanced_dataset_2(file, _engineID):
    """
    Advanced visualizations for a specific engine ID from the training dataset.
    """
    # Charger les données
    _id, cycle, sensor1, sensor2, sensor3, sensor4, ttf, label = load_data("training", file)
    engine_data = _id == _engineID  # Filtrer les données pour l'ID moteur spécifié
    
    # Préparation des données pour le violin plot
    data = [ttf[engine_data & (label == 0)], ttf[engine_data & (label == 1)]]
    
    plt.figure(figsize=(6, 5))
    
    # Violin plot avec des améliorations
    parts = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    
    # Ajouter des couleurs différentes pour chaque violon
    colors = ['lightgreen', 'lightcoral']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])  # Assigner la couleur
        pc.set_edgecolor('black')  # Couleur du bord
        pc.set_alpha(0.8)  # Transparence
    
    # Customiser les lignes pour les médianes et moyennes
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.5)

    # Ajouter des étiquettes sur l'axe des x
    plt.xticks([1, 2], ['Normal (label=0)', 'Faulty (label=1)'])
    
    # Ajouter un titre et des labels pour les axes
    plt.title(f'Time-to-Failure (TTF) vs Label for engine {_engineID}', fontsize=11)
    plt.ylabel('Time-to-Failure (TTF)', fontsize=11)
    
    # Ajouter une grille
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ajuster les marges pour éviter que le titre soit trop proche du graphique
    plt.tight_layout()
    
    plt.show()

plot_advanced_dataset_2(file="train_selected.csv",_engineID=2)


## Corrélation entre les données

# Chargement des données
df = pd.read_csv("train_selected.csv")

# Calcul des corrélations
corrMatrix = df[['s1', 's2', 's3', 's4', 'ttf', 'label_bnc']].corr()
print(corrMatrix)

# Customisation de la heatmap
plt.figure(figsize=(8, 6))

# Palette différente et annotations ajustées
sns.heatmap(corrMatrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
            annot_kws={"size": 11}, linewidths=0.5, linecolor='grey', 
            square=True, cbar_kws={"shrink": 1.0}, 
            xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns)

# Titres et ajustement des axes
plt.title("Correlation Matrix: Sensors and TTF and Label", fontsize=11)
plt.xticks(fontsize=11, rotation=45)
plt.yticks(fontsize=11, rotation=0)
plt.tight_layout()  # Ajuste automatiquement les marges
plt.show()


# Model Selection

# Lecture ou chargement des données
data_train = pd.read_csv("train_selected.csv")
data_test = pd.read_csv("test_selected.csv")

# Récupération des features (capteurs & TTF) et la cible (label)
# Donnée d'entrainement
X = data_train[['s1', 's2', 's3', 's4', 'ttf']].to_numpy()
_label = data_train['label_bnc'].to_numpy()

# Donnée de test
X_test = data_test[['s1', 's2', 's3', 's4']]

# Ajout de la colonne TTF dans les données de test
predicted_ttf = ttf_pred_svr
X_test.insert(4, 'ttf', predicted_ttf)
X_test = X_test.to_numpy()

ttf_test = []

with open("PM_truth.txt") as file:
    content = file.read()
    # diviser la chaine en sous chaine 
    content = content.split(" \n")
    
    # supprimer le dernier element
    content.pop()
    # récupérer le TTF
    for element in content:
        ttf_test.append(int(element))
        
    file.close()

# Assumptions : comme pour les données d'entrainement, pour des ttf > 30 l=0 et ttf < 30 l=1

label_test = []
for i in ttf_test:
    if i>30:
        label_test.append(0)
    else:
        label_test.append(1)

### Séparation des données (train_test_plit)

# Répartition en donnée d'entrainement et de validation
X_train_, X_val_, Y_train, Y_val = train_test_split(X, _label, test_size=0.2, random_state=42)

print(f"Training data size = {X_train_.shape}")
print(f"Validation data size = {X_val_.shape}")

### Normalisation des données

# Nomarlisation des données
## StandardScaler 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train_)
x_val_scaled = scaler.transform(X_val_)
x_test_scaled = scaler.transform(X_test)


# Support Vector Classifier (SVC)

# Initialiser le modèle SVC avec des hyperparamètres manuels
svc_model = SVC(cache_size=1000, random_state=42)
    
# Entraîner le modèle sur les données d'entraînement
svc_model.fit(x_train_scaled, Y_train)

# Evaluer le modèle
score_svc = svc_model.score(x_val_scaled, Y_val)

# Prediction
y_pred_svc = svc_model.predict(x_val_scaled)

print(f"Sensors & TTF (s1, s2, s3, s4, ttf) : {x_val_scaled[0]}")
print(f"Predicted Label : {y_pred_svc[0]}")
print(f"Ground truth Label : {Y_val[0]} \n")

print(f"SVC score : {score_svc} \n")

accuracy = accuracy_score(Y_val, y_pred_svc)
precision = precision_score(Y_val, y_pred_svc)
recall = recall_score(Y_val, y_pred_svc)

print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")


### Fine Tuning

# Initialiser le modèle SVC avec des hyperparamètres manuels
svc_model = SVC(C=10.0, cache_size=1000, random_state=42, gamma='scale')
    
# Entraîner le modèle sur les données d'entraînement
svc_model.fit(x_train_scaled, Y_train)

# Evaluer le modèle
score_ = svc_model.score(x_val_scaled, Y_val)

# Prediction
y_predict_svc = svc_model.predict(x_val_scaled)

print(f"Sensors & TTF (s1, s2, s3, s4, ttf) : {x_val_scaled[0]}")
print(f"Predicted Label : {y_predict_svc[0]}")
print(f"Ground truth Label : {Y_val[0]} \n")

print(f"SVC score : {score_} \n")

accuracy = accuracy_score(Y_val, y_predict_svc)
precision = precision_score(Y_val, y_predict_svc)
recall = recall_score(Y_val, y_predict_svc)

print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")

### Cross-Validation

# Utilisation de Accuracy comme fonction de coût
score_accu = cross_val_score(estimator=svc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Utilisation de Precision comme fonction de coût
score_prec = cross_val_score(estimator=svc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='precision', n_jobs=-1, verbose=1)

# Utilisation de Recall comme fonction de coût
score_rec = cross_val_score(estimator=svc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='recall', n_jobs=-1, verbose=1)

# Affichage des scores Accuracy pour chaque fold
print(f"Accuracy pour chaque fold : {score_accu}")
print(f"Accuracy moyen : {score_accu.mean()}\n")

# Affichage des scores RMSE pour chaque fold
print(f"Precision pour chaque fold : {score_prec}")
print(f"Precision moyen : {score_prec.mean()}\n")

# Affichage des scores RMSE pour chaque fold
print(f"Recall pour chaque fold : {score_rec}")
print(f"Recall moyen : {score_rec.mean()}\n")

# Random Forest Classifier

# Initialiser le modèle RandomForest avec des hyperparamètres
rfc_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=16, n_jobs=-1, verbose=1)

# Entraîner le modèle sur les données d'entraînement
rfc_model.fit(x_train_scaled, Y_train)

# Evaluer le modèle
score_rfc = rfc_model.score(x_val_scaled, Y_val)

# Prediction
y_prediction_rfc = rfc_model.predict(x_val_scaled)

print(f"Sensors (s1, s2, s3, s4) : {x_val_scaled[0]}")
print(f"Predicted Time to Failure : {y_prediction_rfc[0]}")
print(f"Ground truth TTF : {Y_val[0]} \n")

print(f"Random Forest score : {score_rfc} \n")

accuracy = accuracy_score(Y_val, y_prediction_rfc)
precision = precision_score(Y_val, y_prediction_rfc)
recall = recall_score(Y_val, y_prediction_rfc)

print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")


### Cross-Validation
# Utilisation de Accuracy comme fonction de coût
score_accu = cross_val_score(estimator=rfc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Utilisation de Precision comme fonction de coût
score_prec = cross_val_score(estimator=rfc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='precision', n_jobs=-1, verbose=1)

# Utilisation de Recall comme fonction de coût
score_rec = cross_val_score(estimator=rfc_model, X=x_train_scaled, y=Y_train, cv=5, scoring='recall', n_jobs=-1, verbose=1)

# Affichage des scores Accuracy pour chaque fold
print(f"Accuracy pour chaque fold : {score_accu}")
print(f"Accuracy moyen : {score_accu.mean()}\n")

# Affichage des scores RMSE pour chaque fold
print(f"Precision pour chaque fold : {score_prec}")
print(f"Precision moyen : {score_prec.mean()}\n")

# Affichage des scores RMSE pour chaque fold
print(f"Recall pour chaque fold : {score_rec}")
print(f"Recall moyen : {score_rec.mean()}\n")


## Prédiction avec les données de test

### SVC
label_pred_svc = svc_model.predict(x_test_scaled)

accuracy = accuracy_score(label_test, label_pred_svc)
precision = precision_score(label_test, label_pred_svc)
recall = recall_score(label_test, label_pred_svc)

print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")

# Confusion matrix
matrix = confusion_matrix(label_test, label_pred_svc, labels=svc_model.classes_)
disp = ConfusionMatrixDisplay(matrix, display_labels=svc_model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

### Random Forest Classifier

label_pred = rfc_model.predict(x_test_scaled)

accuracy = accuracy_score(label_test, label_pred)
precision = precision_score(label_test, label_pred)
recall = recall_score(label_test, label_pred)

print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")

# Confusion matrix
conf_matrix = confusion_matrix(label_test, label_pred, labels=rfc_model.classes_)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=rfc_model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
