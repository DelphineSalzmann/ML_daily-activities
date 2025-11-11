# %% ------------------------------------------------------------------
# 1. IMPORTATION DES BIBLIOTHÈQUES
# --------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from Create_data import load_all_data

# Importations spécifiques de Scikit-learn pour le pré-traitement et l'évaluation
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer # Pour gérer les données manquantes (NaN)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,  # Pour tracer les courbes ROC
    auc,        # Pour calculer l'aire sous la courbe
    roc_auc_score # Pour calculer le score AUC
)

# Importations spécifiques de Keras (l'API de haut niveau de TensorFlow)
from tensorflow import keras
import keras
from keras import layers

# Affiche les versions pour le débogage
print(f"TensorFlow Version: {tf.__version__}")

# %% ------------------------------------------------------------------
# 2. CONFIGURATION DU SCRIPT
# --------------------------------------------------------------------
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


# %% ------------------------------------------------------------------
# 4. CHARGEMENT DES DONNÉES (AVEC CACHE)
# --------------------------------------------------------------------
# Cette section évite de tout re-calculer à chaque exécution.

# Vérifier si un fichier de données prétraitées existe déjà
if os.path.exists(PROCESSED_DATA_FILE):
    # Si oui, charger les données depuis ce fichier (rapide)
    print(f"Chargement des données en cache depuis '{PROCESSED_DATA_FILE}'...")
    with np.load(PROCESSED_DATA_FILE) as data:
        X_raw = data['x']
        y = data['y']
    print("Chargement depuis le cache terminé.")
    
else:
    # Si non, exécuter le long processus de chargement/extraction
    print(f"Fichier de cache '{PROCESSED_DATA_FILE}' non trouvé.")
    X_raw, y = load_all_data(DATA_BASE_PATH, NUM_ACTIVITIES, NUM_SUBJECTS)
    
    # Sauvegarder les tableaux X et y dans un fichier compressé .npz
    print(f"Sauvegarde des données prétraitées dans '{PROCESSED_DATA_FILE}'...")
    np.savez_compressed(PROCESSED_DATA_FILE, x=X_raw, y=y)
    print("Sauvegarde terminée.")

# Définition des constantes globales basées sur les données chargées
num_features = X_raw.shape[1]
num_classes = NUM_ACTIVITIES
print(f"Données prêtes : {X_raw.shape[0]} échantillons, {num_features} features, {num_classes} classes.")

# %% ------------------------------------------------------------------
# 5. DÉFINITION DU MODÈLE ANN (MLP)
# --------------------------------------------------------------------

def create_model():
    """
    Définit l'architecture de notre Perceptron Multi-Couches (MLP).
    C'est un modèle "FeedForward" (Chap. 3, slide 37).
    
    Nous utilisons les hyperparamètres optimaux trouvés lors d'un
    'Keras Tuner' précédent (voir log) pour de bonnes performances.
    """
    model = keras.Sequential(
        [
            # Couche d'entrée : spécifie la forme des données (180 features)
            layers.Input(shape=(num_features,), name="input_layer"),
            
            # 1ère couche cachée : 128 neurones, activation ReLU
            # ReLU (Chap. 3, slide 10) est rapide et efficace
            layers.Dense(128, activation="relu", name="hidden_layer_1"),
            
            # Couche de Dropout : "éteint" 10% des neurones aléatoirement
            # C'est une technique de régularisation pour éviter le sur-apprentissage.
            layers.Dropout(0.1),
            
            # 2ème couche cachée : 128 neurones, activation ReLU
            layers.Dense(128, activation="relu", name="hidden_layer_2"),
            layers.Dropout(0.4), # Un dropout plus élevé ici
            
            # Couche de sortie : 19 neurones (1 par classe)
            # Fonction d'activation 'softmax' (Chap. 3, slide 22)
            # Elle transforme les sorties en un vecteur de probabilités (somme=1)
            layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ],
        name="mlp_activity_classifier",
    )

    # Compilation du modèle : définition de l'optimiseur et de la fonction de perte
    model.compile(
        # 'adam' est un optimiseur de gradient stochastique efficace
        # 0.0001 était le meilleur taux d'apprentissage (learning_rate) trouvé
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        
        # 'sparse_categorical_crossentropy' est la fonction de perte
        # standard pour la classification multi-classe lorsque les étiquettes (y)
        # sont des entiers (0, 1, 2...) et non des vecteurs one-hot.
        loss='sparse_categorical_crossentropy',
        
        # Métrique à surveiller pendant l'entraînement
        metrics=['accuracy']
    )
    return model

# Affiche un résumé de l'architecture du modèle
model_summary = create_model()
model_summary.summary()

# %% ------------------------------------------------------------------
# 6. VALIDATION CROISÉE (Cross-Validation)
# --------------------------------------------------------------------
print("\n--- Démarrage de la Validation Croisée (Cross-Validation) ---")

# StratifiedKFold garantit que la proportion de chaque classe est
# préservée dans chaque "fold". C'est crucial pour l'évaluation.
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_no = 1

# Boucle sur chaque "fold" (pli)
# `skf.split` génère des indices pour les ensembles d'entraînement et de validation
for train_index, val_index in skf.split(X_raw, y):
    print(f"\n--- Fold {fold_no}/{N_SPLITS_CV} ---")
    
    # 1. Séparer les données pour ce fold
    X_train_cv, X_val_cv = X_raw[train_index], X_raw[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]

    # 2. Mise à l'échelle (Scaling) - ÉTAPE CRUCIALE
    # Nous ajustons le scaler (calcul de la moyenne et de l'écart-type)
    # UNIQUEMENT sur les données d'entraînement (X_train_cv)
    # pour éviter toute fuite d'information (data leakage).
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    
    # Nous APPLIQUONS ensuite cette même transformation (avec la moyenne/std
    # de l'entraînement) aux données de validation.
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)

    # 3. Créer et entraîner un NOUVEAU modèle pour ce fold
    # (Réinitialise les poids à chaque fois)
    model_cv = create_model()
    
    print(f"Entraînement du Fold {fold_no}...")
    model_cv.fit(
        X_train_cv_scaled,
        y_train_cv,
        epochs=50, # 50 époques suffisent pour la validation
        batch_size=64,
        validation_data=(X_val_cv_scaled, y_val_cv),
        # Arrêt anticipé si la performance de validation ne s'améliore pas
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0 # Mode silencieux pour ne pas surcharger le log
    )

    # 4. Évaluer le fold
    loss_cv, accuracy_cv = model_cv.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
    print(f"Fold {fold_no} - Précision de Validation: {accuracy_cv:.4f}")
    fold_accuracies.append(accuracy_cv)

    fold_no += 1

# Afficher les résultats finaux de la validation croisée
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
print("\n--- Résultats de la Validation Croisée ---")
print(f"Précisions individuelles des folds: {[round(f, 4) for f in fold_accuracies]}")
print(f"Précision Moyenne (Accuracy): {mean_accuracy:.4f}")
print(f"Écart-type (Stabilité): {std_accuracy:.4f}")

# %% ------------------------------------------------------------------
# 7. ENTRAÎNEMENT FINAL ET ÉVALUATION SUR LE TEST SET
# --------------------------------------------------------------------
# Maintenant que nous avons validé notre *méthode*, nous entraînons un
# modèle final sur l'ensemble des données d'entraînement (80% du total)
# pour l'évaluer sur l'ensemble de test (20% mis de côté).

print("\n--- Entraînement Final sur 80% des données ---")

# 1. Division Train/Test (80% / 20%)
# Le 'test_set' ici est notre ensemble d'évaluation final,
# il ne doit jamais être utilisé pour l'entraînement.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 2. Mise à l'échelle (fit sur train, transform sur train et test)
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full)
X_test_scaled = scaler_final.transform(X_test)
print(f"Taille du set d'entraînement final: {X_train_full_scaled.shape}")
print(f"Taille du set de test final: {X_test_scaled.shape}")

# 3. Créer le modèle final
final_model = create_model()

# 4. Entraîner le modèle final
history_final = final_model.fit(
    X_train_full_scaled,
    y_train_full,
    epochs=60, # On peut entraîner un peu plus longtemps
    batch_size=64,
    # *** CORRECTION DE LA FUITE DE DONNÉES ***
    # Nous utilisons 10% de nos données d'ENTRAÎNEMENT pour la validation,
    # et non le jeu de test.
    validation_split=0.1, 
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1 # Afficher la progression
)

# %% ------------------------------------------------------------------
# 8. ÉVALUATION FINALE
# --------------------------------------------------------------------
print("\n--- Évaluation Finale sur le Test Set (Données Inconnues) ---")

# Évaluation finale sur le jeu de test, que le modèle n'a jamais vu
loss_test, accuracy_test = final_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Précision (Accuracy) Finale sur le Test Set: {accuracy_test:.4f}")

# Prédictions pour la matrice de confusion et les courbes ROC
y_pred_probs_test = final_model.predict(X_test_scaled)
# Choisir la classe avec la probabilité la plus haute
y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1) 

# --- 8a. Matrice de Confusion ---
# (Comme vu au Chap. 4)
cm_test = confusion_matrix(y_test, y_pred_classes_test)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"A{i+1}" for i in range(num_classes)], 
            yticklabels=[f"A{i+1}" for i in range(num_classes)])
plt.title('Matrice de Confusion Finale sur le Test Set')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes (Labels)')
plt.show()

# --- 8b. Rapport de Classification ---
# Affiche Précision, Rappel, F1-Score (vus au Chap. 4 & 7)
print("\nRapport de Classification Final sur le Test Set:\n")
target_names = [f'Activity {i+1}' for i in range(num_classes)]
print(classification_report(y_test, y_pred_classes_test, target_names=target_names))

# %% ------------------------------------------------------------------
# 8c. COURBES ROC ET SCORE AUC (Multi-classe / One-vs-Rest)
# --------------------------------------------------------------------
print("\n--- Calcul des Courbes ROC et Scores AUC (One-vs-Rest) ---")

# Pour la ROC multi-classe, nous avons besoin des étiquettes de test en
# format binarisé (One-Hot Encoding), shape = (n_samples, 19)
y_test_bin = label_binarize(y_test, classes=range(num_classes))

# y_pred_probs_test a déjà les bonnes probabilités (scores)
y_pred_scores = y_pred_probs_test 

# --- Calcul des scores AUC ---
# "macro": Calcule l'AUC pour chaque classe et fait la moyenne
auc_macro = roc_auc_score(y_test_bin, y_pred_scores, average="macro", multi_class="ovr")
# "weighted": Comme "macro", mais pondère la moyenne par le nombre d'échantillons
auc_weighted = roc_auc_score(y_test_bin, y_pred_scores, average="weighted", multi_class="ovr")

print(f"Score AUC (Moyenne Macro): {auc_macro:.4f}")
print(f"Score AUC (Moyenne Pondérée): {auc_weighted:.4f}")

# --- Calcul des courbes ROC "Micro" et "Macro" pour le graphique ---

# Courbe ROC "Micro-moyennée"
# Aplatit toutes les étiquettes et les scores en un seul vecteur binaire
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_scores.ravel())
auc_val_micro = auc(fpr_micro, tpr_micro)

# Courbe ROC "Macro-moyennée"
# 1. Collecter les FPR/TPR pour chaque classe
all_fpr = []
all_tpr = []
for i in range(num_classes):
    fpr_class, tpr_class, _ = roc_curve(y_test_bin[:, i], y_pred_scores[:, i])
    all_fpr.append(fpr_class)
    all_tpr.append(tpr_class)

# 2. Créer une grille de FPR commune (en interpolant)
all_fpr_unique = np.unique(np.concatenate([all_fpr[i] for i in range(num_classes)]))

# 3. Interpoler toutes les courbes TPR sur cette grille
mean_tpr = np.zeros_like(all_fpr_unique)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr_unique, all_fpr[i], all_tpr[i])

# 4. Moyennage et calcul de l'AUC macro
mean_tpr /= num_classes
fpr_macro = all_fpr_unique
tpr_macro = mean_tpr
auc_val_macro = auc(fpr_macro, tpr_macro) 

# --- Tracer les courbes ROC ---
plt.figure(figsize=(10, 8))
plt.plot(fpr_micro, tpr_micro,
         label=f'ROC Micro-moyennée (AUC = {auc_val_micro:0.3f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr_macro, tpr_macro,
         label=f'ROC Macro-moyennée (AUC = {auc_val_macro:0.3f})',
         color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)') # Ligne de chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbes ROC Multi-classe (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %% ------------------------------------------------------------------
# 8d. COURBES D'APPRENTISSAGE
# --------------------------------------------------------------------
# Afficher les courbes de perte et de précision de l'entraînement final
plt.figure(figsize=(12, 5))

# Graphique de Précision
plt.subplot(1, 2, 1)
plt.plot(history_final.history['accuracy'], label='Précision (Entraînement)')
plt.plot(history_final.history['val_accuracy'], label='Précision (Validation)')
plt.title('Courbe de Précision (Accuracy)')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.grid(True)

# Graphique de Perte
plt.subplot(1, 2, 2)
plt.plot(history_final.history['loss'], label='Perte (Entraînement)')
plt.plot(history_final.history['val_loss'], label='Perte (Validation)')
plt.title('Courbe de Perte (Loss)')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Fin du script.")