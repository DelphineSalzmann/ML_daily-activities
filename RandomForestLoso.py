## --Importer les bibliothèques nécessaires --
import os
import numpy as np
import time

# Importer la fonction de création de données
from Create_data import load_all_data

# Bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, cross_val_score, GridSearchCV, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
# --- Configuration ---
# Paramètres basés sur la description du dataset (README.md)
NUM_ACTIVITIES = 19
NUM_SUBJECTS = 8
NUM_SEGMENTS_PER_SUBJECT = 60
NUM_SENSORS = 45
SAMPLES_PER_SEGMENT = 125 # 5 sec * 25 Hz = 125 échantillons par segment

config={'features':180}

# Chemin vers le dossier de données (relatif au script)
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data')
# Fichier pour stocker les données prétraitées et accélérer les lancements futurs
if config['features'] == 90:
    PROCESSED_DATA_FILE = 'processed_data_90features.npz'
else:
    PROCESSED_DATA_FILE = 'processed_data_180features.npz'

# Paramètres pour le Machine Learning
N_SPLITS_CV = 5 # Nombre de "folds" (plis) pour la validation croisée
RANDOM_STATE = 42 # Garantit que les divisions aléatoires sont reproductibles


# --- Charger les données ---

if os.path.exists(PROCESSED_DATA_FILE):
    print(f"Chargement des données en cache depuis '{PROCESSED_DATA_FILE}'...")
    with np.load(PROCESSED_DATA_FILE) as data:
        X_raw = data['x']
        y = data['y']
        groups = data['groups']   
    print("Chargement depuis le cache terminé.")
else:
    # Si le cache n'existe pas, exécuter le long processus de chargement
    print(f"Fichier de cache '{PROCESSED_DATA_FILE}' non trouvé.")
    print("Exécution du chargement complet (cela peut prendre 1-2 minutes)...")
    X_raw, y, groups = load_all_data(DATA_BASE_PATH, NUM_ACTIVITIES, NUM_SUBJECTS)
    
    print(f"Sauvegarde des données prétraitées dans '{PROCESSED_DATA_FILE}'...")
    np.savez_compressed(PROCESSED_DATA_FILE, x=X_raw, y=y, groups=groups)
    print("Sauvegarde terminée.")

num_features = X_raw.shape[1]
num_classes = NUM_ACTIVITIES

print(f"Final shape of X after imputation: {X_raw.shape}"
      f"\nNumber of features: {num_features}"
      f"\nNumber of classes: {num_classes}")

# --- Division Train/Test par la méthode LOSO : on exclu une personne de l'entraînement et on teste sur elle ---

def Train_Test(exclu):
    train_idx = np.where(groups != exclu)[0]
    test_idx  = np.where(groups == exclu)[0]
    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

# --- 3. Modèle de base ---

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)

# --- Déterminer les meilleurs paramètres pour le modèle ---

def best_model():
    param_grid = {
        'n_estimators': [100,200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # --- Leave-One-Subject-Out cross-validation ---
    cv = LeaveOneGroupOut()

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_raw, y, groups=groups)

    print("\n--- Résultats de la recherche d'hyperparamètres (LOSO) ---")
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure accuracy moyenne (LOSO) :", grid_search.best_score_)

    return grid_search


# grid_search = best_model()
# best_rf = grid_search.best_estimator_
# print(best_rf)

# ou bien, une fois qu'on a déjà les meilleurs paramètres :
best_rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= True, max_depth= None, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 200)

## -- Vallidation croisée LOSO et matrice de confusion ---

def validation_LOSO():

    logo = LeaveOneGroupOut()

    accuracies=[]
    y_true_all = []
    y_pred_all = []
    y_score_all = [] #pour ROC et Precision-Recall
    target_classes = np.arange(0, 19)

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        print(f"--- Test sur groupe {groups[test_idx][0]} ---")

        # Séparation
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Nouveau modèle à chaque itération
        model = best_rf

        # Entraînement
        debut=time.time()
        model.fit(X_train, y_train)
        fin=time.time()

        # Prédiction
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        reorder_idx = [list(model.classes_).index(c) for c in target_classes]
        y_score = y_score[:, reorder_idx]

        # Stockage des résultats
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        y_score_all.append(y_score)

        print("  Accuracy groupe :", acc)
        print("Temps d'entraînement", fin-debut)

    accuracies = np.array(accuracies)
    acc_mean = accuracies.mean()
    acc_std = accuracies.std()
    y_score_all = np.vstack(y_score_all)

    # --- Matrice de confusion finale sur tous les groupes ---
    cm = confusion_matrix(y_true_all, y_pred_all)

    print("\nAccuracy globale :", acc_mean)
    print("\nEcart-type accuracy :", acc_std)

    return cm, np.array(y_true_all),y_score_all

def plot_confusion(cm):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"A{i+1}" for i in range(num_classes)], 
                yticklabels=[f"A{i+1}" for i in range(num_classes)])
    plt.title('Matrice de Confusion Finale sur le Test Set')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes (Labels)')
    plt.show()

# cm, y_true_all, y_score_all=validation_LOSO()
# print(cm)
# plot_confusion(cm)

###--- Impureté de Gini ---

def foret_gini(trained_rf):
    tree_ginis = []
    for tree in trained_rf.estimators_:
        impurity = tree.tree_.impurity
        n_samples = tree.tree_.n_node_samples
        weighted_gini = np.sum(impurity * n_samples) / np.sum(n_samples)
        tree_ginis.append(weighted_gini)
    return tree_ginis

def plot_gini(rf):
    tree_ginis=foret_gini(rf)
    plt.figure(figsize=(10,6))
    plt.bar(range(1, len(tree_ginis)+1), tree_ginis, color='skyblue')
    plt.axhline(np.mean(tree_ginis), color='red', linestyle='--', label=f"Moyenne forêt = {np.mean(tree_ginis):.4f}")
    plt.xlabel("Arbre n°")
    plt.ylabel("Impureté de Gini moyenne pondérée")
    plt.title("Impureté de Gini moyenne pondérée de chaque arbre dans la forêt")
    plt.legend()
    plt.show()
    

#X_train,X_test,y_train,y_test=Train_Test(2)
#best_rf.fit(X_train,y_train)
#plot_gini(best_rf)


##--- Effet de 3 paramètres : nombre d'arbres, nombre min d'échantillons par feuille et profondeur max ---
def Influence_n_estimators_LOSO(sujets_exclus):

    n_estimators_options = [10, 50, 100, 200, 300, 400, 500]

    logo = LeaveOneGroupOut()

    train_acc_means = []
    test_acc_means = []
    gini_means = []

    # Pour chaque valeur de n_estimators
    for n in n_estimators_options:
        print("\nRésultats pour ", n,"arbres")
        train_accs = []
        test_accs = []

        # LOSO
        for sujet in sujets_exclus:
            print("\nTest sur le sujet", sujet)
            X_train, X_test ,y_train,y_test=Train_Test(sujet)

            rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= True, max_depth= None, min_samples_leaf= 2, min_samples_split= 2, n_estimators= n)
            rf.fit(X_train, y_train)

            # Accuracies
            train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
            test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

            # Gini
            tree_ginis=foret_gini(rf)

        # Moyennes LOSO
        train_acc_means.append(np.mean(train_accs))
        test_acc_means.append(np.mean(test_accs))
        gini_means.append(np.mean(tree_ginis))

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_estimators_options, train_acc_means, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_estimators_options, test_acc_means, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Nombre d\'arbres')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_estimators_options, gini_means, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle("Impureté des arbres vs Accuracy en fonction du nombre d'arbres (LOSO)")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    plt.show()

def Influence_nleaf_LOSO(sujets_exclus):

    n_min_samples_leaf_options = [1,2,5,10,50,100]

    logo = LeaveOneGroupOut()

    train_acc_means = []
    test_acc_means = []
    gini_means = []

    # Pour chaque valeur de n_estimators
    for n in n_min_samples_leaf_options:
        print("\nRésultats pour ", n,"as minimal samples leaf")
        train_accs = []
        test_accs = []


        # LOSO
        for sujet in sujets_exclus:
            print("\nTest sur le sujet", sujet)
            X_train, X_test, y_train,y_test = Train_Test(sujet)

            rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= True, max_depth= None, min_samples_leaf= n, min_samples_split= 2, n_estimators= 200)
            rf.fit(X_train, y_train)

            # Accuracies
            train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
            test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

            # Gini
            tree_ginis=foret_gini(rf)

        # Moyennes LOSO
        train_acc_means.append(np.mean(train_accs))
        test_acc_means.append(np.mean(test_accs))
        gini_means.append(np.mean(tree_ginis))

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_min_samples_leaf_options, train_acc_means, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_min_samples_leaf_options, test_acc_means, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Nombre d\'échantillon minimal par feuille')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_min_samples_leaf_options, gini_means, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle("Impureté des arbres vs Accuracy en fonction du nombre minimal d'échantillons par feuille (LOSO)")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    plt.show()

def Influence_max_depth_LOSO(sujets_exclus):

    n_max_depth_options = [5,10,20,30]

    logo = LeaveOneGroupOut()

    train_acc_means = []
    test_acc_means = []
    gini_means = []

    # Pour chaque valeur de n_estimators
    for n in n_max_depth_options:
        train_accs = []
        test_accs = []

        # LOSO
        for sujet in sujets_exclus:
            print("\nTest sur le sujet", sujet)
            X_train, X_test, y_train,y_test = Train_Test(sujet)

            rf = RandomForestClassifier(random_state=RANDOM_STATE, bootstrap= True, max_depth= None, min_samples_leaf= n, min_samples_split= 2, n_estimators= 200)
            rf.fit(X_train, y_train)

            # Accuracies
            train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
            test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

            # Gini
            tree_ginis=foret_gini(rf)

        # Moyennes LOSO
        train_acc_means.append(np.mean(train_accs))
        test_acc_means.append(np.mean(test_accs))
        gini_means.append(np.mean(tree_ginis))

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    color1 = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(n_max_depth_options, train_acc_means, color=color1, linestyle='--', label='Train Accuracy')
    ax1.plot(n_max_depth_options, test_acc_means, color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_xlabel('Profondeur maximale des arbres')
    ax2.set_ylabel('Impureté moyenne pondérée de Gini', color=color2)
    ax2.plot(n_max_depth_options, gini_means, color=color2, label='Impureté moyenne')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle("Impureté des arbres vs Accuracy en fonction de la profondeur des arbres (LOSO)")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    plt.show()



#Influence_n_estimators_LOSO([1,2,3,4,5,6,7,8])
#Influence_nleaf_LOSO([1,2,3,4,5,6,7,8])
#Influence_max_depth_LOSO([1,2,3,4,5,6,7,8])

def plot_ROC(y_true_all, y_score_all):
 
    # --- noms des classes ---
    n_classes = 19
    class_names = [f"A{i+1}" for i in range(n_classes)]  # juste pour l'affichage
    y_true_bin = label_binarize(y_true_all, classes=np.arange(n_classes))

    plt.figure(figsize=(12, 10))

    # --- ROC curve par classe ---
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_all[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)  # ligne diagonale
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbes ROC par classe (19 activités)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.show()

def plot_PR(y_true_all, y_score_all):

    # --- noms des classes ---
    n_classes = 19
    class_names = [f"A{i+1}" for i in range(n_classes)]  # juste pour l'affichage
    y_true_bin = label_binarize(y_true_all, classes=np.arange(n_classes))

    plt.figure(figsize=(12, 10))

    # --- PR curve par classe ---
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_all[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score_all[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbes Précision–Rappel")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True)
    plt.show()

cm, y_true_all,y_score_all = validation_LOSO()
print(y_score_all.shape)   # doit être (N_total, 19)
print(y_score_all[:5,:])   # probabilités entre 0 et 1
print(y_true_all[:5]) 
#print(y_true_bin[:5,:])

plot_ROC(y_true_all,y_score_all)
plot_PR(y_true_all,y_score_all)