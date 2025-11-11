## --Importer les bibliothèques nécessaires --
import os
import numpy as np

# Importer la fonction de création de données
from Create_data import load_all_data

# Bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# --- Configuration ---
# Paramètres basés sur la description du dataset (README.md)
NUM_ACTIVITIES = 19
NUM_SUBJECTS = 8
NUM_SEGMENTS_PER_SUBJECT = 60
NUM_SENSORS = 45
SAMPLES_PER_SEGMENT = 125 # 5 sec * 25 Hz = 125 échantillons par segment

# Chemin vers le dossier de données (relatif au script)
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data')
# Fichier pour stocker les données prétraitées et accélérer les lancements futurs
PROCESSED_DATA_FILE = 'processed_data.npz'

# Paramètres pour le Machine Learning
N_SPLITS_CV = 5 # Nombre de "folds" (plis) pour la validation croisée
RANDOM_STATE = 42 # Garantit que les divisions aléatoires sont reproductibles


# ---1. Charger les données ---

X_raw, y = load_all_data(DATA_BASE_PATH, NUM_ACTIVITIES, NUM_SUBJECTS)
num_features = X_raw.shape[1]
num_classes = NUM_ACTIVITIES

print(f"Final shape of X after imputation: {X_raw.shape}"
      f"\nNumber of features: {num_features}"
      f"\nNumber of classes: {num_classes}")



# --- 2. Division Train/Test ---

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


# --- 3. Modèle de base ---

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)


# --- 4. Déterminer les meilleurs paramètres pour le modèle ---

def best_model():
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\n--- Résultats de la recherche d'hyperparamètres ---")
    print("Meilleurs paramètres trouvés :", grid_search.best_params_)
    print("Meilleure accuracy moyenne en CV :", grid_search.best_score_)

    return grid_search

#grid_search = best_model()
#best_rf = grid_search.best_estimator_

# ou bien, une fois qu'on a déjà les meilleurs paramètres :
best_rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= False, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200)


# ## --5. Validation croisée ---

# print("Démarage de la validation croisée")
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
# scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring="accuracy")
# print(scores)
# print("Accuracy moyenne CV :", scores.mean())
# print("Écart type CV :", scores.std())


# ## --6. Entraînement et validation finale sur le jeu de test ---

best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

print("\n--- Évaluation sur le jeu de test ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Comparaison avec accuracy d'entraînement
# train_acc = accuracy_score(y_train, best_rf.predict(X_train))
# test_acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy entraînement : {train_acc:.3f}")
# print(f"Accuracy test : {test_acc:.3f}")

# # --- 7. Matrice de confusion ---
# cm_test = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=[f"A{i+1}" for i in range(num_classes)], 
#             yticklabels=[f"A{i+1}" for i in range(num_classes)])
# plt.title('Matrice de Confusion Finale sur le Test Set')
# plt.xlabel('Prédictions')
# plt.ylabel('Vraies étiquettes (Labels)')
# plt.show()



# #section pour tracer l’importance des features du modèle final 
###--- 8. Impureté de Gini ---

def foret_gini(rf):
    tree_ginis = []
    for tree in rf.estimators_:
        impurity = tree.tree_.impurity
        n_samples = tree.tree_.n_node_samples
        weighted_gini = np.sum(impurity * n_samples) / np.sum(n_samples)
        tree_ginis.append(weighted_gini)

    plt.figure(figsize=(10,6))
    plt.bar(range(1, len(tree_ginis)+1), tree_ginis, color='skyblue')
    plt.axhline(np.mean(tree_ginis), color='red', linestyle='--', label=f"Moyenne forêt = {np.mean(tree_ginis):.4f}")
    plt.xlabel("Arbre n°")
    plt.ylabel("Impureté de Gini moyenne pondérée")
    plt.title("Impureté de Gini moyenne pondérée de chaque arbre dans la forêt")
    plt.legend()
    plt.show()
    

#foret_gini(best_rf)




##--- 9. Effet du nombre d'arbres n_estimators ---
def Influence_n_estimators():
    n_estimators_options = [10, 50, 100, 200, 300, 400, 500]
    train_accuracies = []
    test_accuracies = []
    mean_ginis = []

    for n in n_estimators_options:
        rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= False, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators=n)
        rf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, rf.predict(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        tree_ginis = []
        for tree in rf.estimators_:
            impurity = tree.tree_.impurity
            n_samples = tree.tree_.n_node_samples
            weighted_gini = np.sum(impurity * n_samples) / np.sum(n_samples)
            tree_ginis.append(weighted_gini)
        mean_ginis.append(np.mean(tree_ginis))


    fig, ax1 = plt.subplots(figsize=(12,6))

    
    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_estimators_options, train_accuracies, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_estimators_options, test_accuracies, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)


    # Axe Y secondaire pour accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Arbre n°')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_estimators_options, mean_ginis, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)
    # Titre et légende
    fig.suptitle("Impureté des arbres vs Accuracy en fonction du nombre d'arbres")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    plt.xlabel('Nombre d\'arbres')
    plt.show()

#Influence_n_estimators()

def Influence_nleaf():
    n_min_samples_leaf_options = [1,5,10,50,100,1000]
    train_accuracies = []
    test_accuracies = []
    mean_ginis = []

    for n in n_min_samples_leaf_options:
        rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= False, max_depth= None, min_samples_leaf= n, min_samples_split= 2, n_estimators=200)
        rf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, rf.predict(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        tree_ginis = []
        for tree in rf.estimators_:
            impurity = tree.tree_.impurity
            n_samples = tree.tree_.n_node_samples
            weighted_gini = np.sum(impurity * n_samples) / np.sum(n_samples)
            tree_ginis.append(weighted_gini)
        mean_ginis.append(np.mean(tree_ginis))


    fig, ax1 = plt.subplots(figsize=(12,6))

    
    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_min_samples_leaf_options, train_accuracies, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_min_samples_leaf_options, test_accuracies, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)


    # Axe Y secondaire pour accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Arbre n°')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_min_samples_leaf_options, mean_ginis, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)
    # Titre et légende
    fig.suptitle("Impureté des arbres vs Accuracy en fonction du nombre minimal d'échantillon par feuille")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    
    plt.xlabel('Nombre minimal d\'échantiillon par feuille')
    plt.show()

#Influence_nleaf()

def Influence_max_depth():
    n_max_depth_options = [5,10,20,30, None]
    train_accuracies = []
    test_accuracies = []
    mean_ginis = []

    for n in n_max_depth_options:
        rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= False, max_depth= n, min_samples_leaf= 1, min_samples_split= 2, n_estimators=200)
        rf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, rf.predict(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        tree_ginis = []
        for tree in rf.estimators_:
            impurity = tree.tree_.impurity
            n_samples = tree.tree_.n_node_samples
            weighted_gini = np.sum(impurity * n_samples) / np.sum(n_samples)
            tree_ginis.append(weighted_gini)
        mean_ginis.append(np.mean(tree_ginis))


    fig, ax1 = plt.subplots(figsize=(12,6))

    
    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_max_depth_options, train_accuracies, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_max_depth_options, test_accuracies, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)


    # Axe Y secondaire pour accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Arbre n°')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_max_depth_options, mean_ginis, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)
    # Titre et légende
    fig.suptitle("Impureté des arbres vs Accuracy en fonction de la profondeur maximale autorisée pour les arbres")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    
    plt.xlabel('Pronfondeur maximale des arbres')
    plt.show()

Influence_max_depth()