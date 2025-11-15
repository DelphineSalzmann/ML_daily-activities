# %% ------------------------------------------------------------------
# 1. IMPORTATION DES BIBLIOTHÈQUES
# --------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from collections import Counter
import joblib # Pour sauvegarder le scaler
import matplotlib.pyplot as plt
import seaborn as sns
import time # Ajout pour mesurer le temps d'exécution

# --- Scikit-learn ---
# Pour les stratégies de validation (croisée, LOSO, train/test)
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneGroupOut
# Pour la normalisation (StandardScaler, MinMaxScaler) et l'encodage (label_binarize)
from sklearn.preprocessing import StandardScaler, label_binarize, MinMaxScaler
# Pour la gestion des données manquantes
from sklearn.impute import SimpleImputer
# Pour l'évaluation des performances (cf. Chapitre 4 & 7 des slides)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    f1_score, recall_score, precision_score, accuracy_score, precision_recall_curve
)

# --- TensorFlow/Keras ---
# Pour la construction du Réseau de Neurones Artificiels (cf. Chapitre 3)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Reproductibilité ---
# Fixer les graines aléatoires assure que les résultats (initialisation des poids, 
# divisions de données) sont identiques à chaque exécution.
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")

# %% ------------------------------------------------------------------
# 2. CONFIGURATION DU SCRIPT (PANNEAU DE CONTRÔLE)
# --------------------------------------------------------------------

# --- PANNEAU DE CONTRÔLE POUR L'ANALYSE DE SENSIBILITÉ ---
# Cette variable permet de choisir le scénario de test à exécuter.
# En modifiant SEULEMENT cette ligne, on peut reproduire n'importe quel
# test de notre analyse (comme demandé dans le "Mode d'emploi" du projet).
#
# Scénarios disponibles :
# "CHAMPION"         : Modèle final optimisé (180f, 128x128, D(0.3/0.5), B32, StandardScaler)
# "BASELINE"         : Modèle initial (180f, 128x128, D(0.1/0.4), B64, StandardScaler)
# "TEST_90_FEATURES"   : Test 1.3 (90f, 128x128, D(0.1/0.4), B64, StandardScaler)
# "TEST_SIMPLE_ARCHI"  : Test 2.1 (180f, 64x32, D(0.2/0.2), B64, StandardScaler)
# "TEST_NO_DROPOUT"    : Test 2A (180f, 128x128, D(0.0/0.0), B64, StandardScaler)
# "TEST_MINMAX_SCALER" : Test 4 (180f, 128x128, D(0.1/0.4), B64, MinMaxScaler)
# --------------------------------------------------------
TEST_SCENARIO = "CHAMPION" 
# --------------------------------------------------------

# --- Configuration de base (constantes) ---
NUM_ACTIVITIES = 19
NUM_SUBJECTS = 8
NUM_SEGMENTS_PER_SUBJECT = 60
NUM_SENSORS = 45
SAMPLES_PER_SEGMENT = 125 # 5 sec * 25 Hz = 125 échantillons par segment
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data')
N_SPLITS_CV = 5 
RANDOM_STATE = 42 

# --- Application de la configuration du scénario ---
print(f"--- Exécution du Scénario : {TEST_SCENARIO} ---")

# Dictionnaire de configuration par défaut (basé sur le "CHAMPION")
# C'est la configuration qui maximise la généralisation (score LOSO).
config = {
    'features': 180,
    'architecture': '128x128',
    'dropout_rates': (0.3, 0.5),
    'scaler_class': StandardScaler,
    'batch_size': 32
}

# Surcharge la configuration par défaut en fonction du scénario de test choisi
if TEST_SCENARIO == "BASELINE":
    config['dropout_rates'] = (0.1, 0.4)
    config['batch_size'] = 64
elif TEST_SCENARIO == "TEST_90_FEATURES":
    config['features'] = 90
    config['dropout_rates'] = (0.1, 0.4)
    config['batch_size'] = 64
elif TEST_SCENARIO == "TEST_SIMPLE_ARCHI":
    config['architecture'] = '64x32'
    config['dropout_rates'] = (0.2, 0.2) # Dropout moyen pour un modèle plus petit
    config['batch_size'] = 64
elif TEST_SCENARIO == "TEST_NO_DROPOUT":
    config['dropout_rates'] = (0.0, 0.0)
    config['batch_size'] = 64
elif TEST_SCENARIO == "TEST_MINMAX_SCALER":
    config['scaler_class'] = MinMaxScaler 
    config['dropout_rates'] = (0.1, 0.4)
    config['batch_size'] = 64

# Le nom du fichier cache dépend du nombre de features (pour ne pas mélanger les données)
if config['features'] == 90:
    PROCESSED_DATA_FILE = 'processed_data_90features.npz'
else:
    PROCESSED_DATA_FILE = 'processed_data_180features.npz'

print(f"Config active: {config}")

# %% ------------------------------------------------------------------
# 3. FONCTIONS DE CHARGEMENT ET D'EXTRACTION
# --------------------------------------------------------------------

def load_segments_for_activity_person(base_path, activity_id, person_id):
    """
    Charge les 60 segments (fichiers s01.txt à s60.txt) pour une activité 
    et une personne données. Gère les fichiers manquants ou corrompus.
    """
    folder_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}")
    
    if not os.path.exists(folder_path):
        print(f"AVERTISSEMENT: Dossier non trouvé, ignoré : {folder_path}")
        return None
        
    all_segments_data = []
    for i in range(1, NUM_SEGMENTS_PER_SUBJECT + 1):
        filename = f"s{i:02d}.txt"
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            segment_data = np.loadtxt(file_path, delimiter=',')
            # Vérification critique de la forme (shape) des données
            if segment_data.shape == (SAMPLES_PER_SEGMENT, NUM_SENSORS):
                all_segments_data.append(segment_data)
            else:
                print(f"ERREUR: Forme incorrecte dans {filename}. Attendu {(SAMPLES_PER_SEGMENT, NUM_SENSORS)}, Reçu {segment_data.shape}")
        except Exception as e:
            print(f"ERREUR: Impossible de charger {filename}. Détail : {str(e)}")
    
    return all_segments_data if all_segments_data else None

def extract_features(segment_data):
    """
    Transforme un segment temporel (125, 45) en un vecteur de caractéristiques.
    C'est l'étape d'Ingénierie des Caractéristiques (Feature Engineering).
    Le nombre de features (90 ou 180) dépend de la config globale.
    """
    features = []
    
    # Gérer les segments manquants (None)
    if segment_data is None:
        num_stats = 2 if config['features'] == 90 else 4
        return np.full(NUM_SENSORS * num_stats, np.nan)

    # Boucle sur chacun des 45 capteurs (colonnes)
    for sensor_col in range(NUM_SENSORS):
        sensor_signal = segment_data[:, sensor_col]
        
        # Caractéristiques de base (toujours incluses)
        features.append(np.mean(sensor_signal))
        features.append(np.std(sensor_signal))
        
        # Ajoute min/max UNIQUEMENT si le scénario de test le demande
        if config['features'] == 180:
            features.append(np.min(sensor_signal))
            features.append(np.max(sensor_signal))
            
    return np.array(features)

def load_all_data(base_path, num_activities, num_subjects):
    """
    Orchestre le chargement de tous les fichiers, l'extraction de caractéristiques
    et l'imputation des données manquantes.
    """
    all_features = []
    all_labels = []
    all_groups = []   # Stocke l'ID du sujet (person_id) pour la CV LOSO
    print("Démarrage du chargement des données et de l'extraction de caractéristiques...")
    
    for activity_label in range(num_activities):
        for person_id in range(1, num_subjects + 1):
            segments = load_segments_for_activity_person(
                base_path, activity_label + 1, person_id
            )
            
            if segments:
                for segment in segments:
                    features = extract_features(segment)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(activity_label)
                        all_groups.append(person_id) # Enregistrer le sujet

    print(f"Chargement terminé. {len(all_features)} segments trouvés.")
    
    if not all_features:
        raise ValueError("Aucune caractéristique n'a été extraite. Vérifiez les chemins et la configuration.")

    X = np.array(all_features)
    y = np.array(all_labels)
    groups = np.array(all_groups)    
    
    # --- Gestion des Données Manquantes (NaN) ---
    print("Gestion des valeurs manquantes (imputation)...")
    nan_count = np.isnan(X).sum()
    print(f"Nombre de NaN avant imputation: {nan_count}")

    if nan_count > 0:
        # Remplace tous les NaN par la moyenne de leur colonne (feature)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print(f"Nombre de NaN après imputation: {np.isnan(X).sum()}")
    else:
        print("Aucune valeur NaN détectée.")

    return X, y, groups   

# %% ------------------------------------------------------------------
# 4. CHARGEMENT DES DONNÉES (AVEC CACHE)
# --------------------------------------------------------------------
# Évite de recalculer l'extraction de features (très long) à chaque exécution.
# Le nom du fichier cache est défini dynamiquement dans la Section 2.

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

# Définition des constantes globales basées sur les données chargées
num_features = X_raw.shape[1]
num_classes = NUM_ACTIVITIES
print(f"Données prêtes : {X_raw.shape[0]} échantillons, {num_features} features, {num_classes} classes.")

# Vérifier la distribution des classes (équilibrée) et des sujets
print("Distribution des classes (label: count):", dict(Counter(y)))
print("Distribution des sujets (subject_id: count):", dict(sorted(Counter(groups).items())))
print(f"Nombre total de sujets : {len(np.unique(groups))}")

# %% ------------------------------------------------------------------
# 5. DÉFINITION DU MODÈLE ANN (MLP)
# --------------------------------------------------------------------

def create_model():
    """
    Définit l'architecture de notre Perceptron Multi-Couches (MLP) [cf. Chapitre 3].
    Les paramètres (architecture, dropout) sont lus depuis la 'config' globale
    pour permettre une analyse de sensibilité facile.
    """
    
    # Lire l'architecture depuis la config
    if config['architecture'] == '64x32':
        units_1, units_2 = 64, 32
    else: # Par défaut ou "128x128"
        units_1, units_2 = 128, 128
        
    # Lire les taux de dropout depuis la config
    dropout_1, dropout_2 = config['dropout_rates']

    model = keras.Sequential(
        [
            # Couche d'entrée : spécifie la forme des données (90 ou 180 features)
            layers.Input(shape=(num_features,), name="input_layer"),
            
            # 1ère couche cachée (Dense = Full-Connected)
            layers.Dense(units_1, activation="relu", name="hidden_layer_1"),
            # Couche de régularisation Dropout. Un taux de 0.0 est ignoré par Keras.
            layers.Dropout(dropout_1), 
            
            # 2ème couche cachée
            layers.Dense(units_2, activation="relu", name="hidden_layer_2"),
            layers.Dropout(dropout_2), 
            
            # Couche de sortie : 19 neurones (1 par classe)
            # Activation 'softmax' pour la classification multi-classe (cf. Chapitre 3, slide 22)
            layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ],
        name=f"mlp_arch_{config['architecture']}_drop_{dropout_1}_{dropout_2}",
    )

    # Compilation du modèle
    model.compile(
        # Optimiseur Adam avec un learning rate fixe, identifié comme optimal.
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        
        # Fonction de perte pour classification multi-classe avec labels entiers (0, 1, ... 18)
        loss='sparse_categorical_crossentropy',
        
        # Métrique à surveiller
        metrics=['accuracy']
    )
    return model

# Affiche un résumé de l'architecture du modèle choisi
model_summary = create_model()
model_summary.summary()

# %% ------------------------------------------------------------------
# 6. VALIDATION CROISÉE "MIXTE" (StratifiedKFold)
# --------------------------------------------------------------------
print("\n--- Démarrage de la Validation Croisée (StratifiedKFold) ---")

# StratifiedKFold garantit que la proportion de chaque classe (activité)
# est préservée dans chaque "fold".
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_no = 1
start_time_cv = time.time() # Mesure du temps

for train_index, val_index in skf.split(X_raw, y):
    print(f"\n--- Fold {fold_no}/{N_SPLITS_CV} ---")
    
    # 1. Séparer les données pour ce fold
    X_train_cv, X_val_cv = X_raw[train_index], X_raw[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]

    # 2. Mise à l'échelle (Scaling)
    # Le scaler est ajusté (fit) UNIQUEMENT sur les données d'entraînement (X_train_cv)
    # pour éviter la fuite de données (data leakage).
    scaler_cv = config['scaler_class']()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    # Le scaler est appliqué (transform) aux données de validation.
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)

    # 3. Créer et entraîner un NOUVEAU modèle (poids réinitialisés)
    model_cv = create_model()
    
    print(f"Entraînement du Fold {fold_no}...")
    model_cv.fit(
        X_train_cv_scaled,
        y_train_cv,
        epochs=50, # 50 époques suffisent pour la validation
        batch_size=config['batch_size'], # Paramètre de la config
        validation_data=(X_val_cv_scaled, y_val_cv),
        # Arrêt anticipé si la val_loss ne s'améliore pas pendant 10 époques
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0 # Mode silencieux
    )

    # 4. Évaluer le fold
    loss_cv, accuracy_cv = model_cv.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
    print(f"Fold {fold_no} - Précision de Validation: {accuracy_cv:.4f}")
    fold_accuracies.append(accuracy_cv)
    fold_no += 1

end_time_cv = time.time()
print(f"\n--- Résultats de la Validation Croisée (Stratifiée) ---")
print(f"Précisions individuelles des folds: {[round(f, 4) for f in fold_accuracies]}")
print(f"Précision Moyenne (Accuracy): {np.mean(fold_accuracies):.4f}")
print(f"Écart-type (Stabilité): {np.std(fold_accuracies):.4f}")
print(f"Temps total CV: {end_time_cv - start_time_cv:.2f} secondes")

# %% ------------------------------------------------------------------
# 6b. VALIDATION "RÉALISTE" (Leave-One-Subject-Out)
# --------------------------------------------------------------------
print("\n--- Validation Leave-One-Subject-Out (LOSO) ---")
# LOGO est le test de généralisation inter-sujet.
logo = LeaveOneGroupOut()

# Listes pour agréger les résultats de tous les folds
y_val_all_loso = []
y_pred_all_loso = []
y_score_all_loso = [] 
fold_results_loso = {'accuracy': [], 'f1_macro': []}
fold_no = 1
start_time_loso = time.time() # Mesure du temps

# `logo.split` utilise `groups` pour s'assurer que l'entraînement (train_idx)
# et la validation (val_idx) ne contiennent jamais le même sujet.
for train_idx, val_idx in logo.split(X_raw, y, groups):
    subject_id = groups[val_idx][0]
    print(f"\n--- Sujet tenu hors entraînement : fold {fold_no} (Sujet {subject_id}) ---")
    X_train_cv, X_val_cv = X_raw[train_idx], X_raw[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]

    # Le scaling est aussi fait dans la boucle LOSO
    scaler_cv = config['scaler_class']()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)

    model_cv = create_model()
    model_cv.fit(X_train_cv_scaled, y_train_cv,
                 epochs=50, 
                 batch_size=config['batch_size'], # Paramètre de la config
                 validation_data=(X_val_cv_scaled, y_val_cv),
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                 verbose=0)

    # Prédictions (probas) et classes prédites
    y_pred_proba = model_cv.predict(X_val_cv_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Agrégation pour la matrice de confusion LOSO globale
    y_val_all_loso.extend(y_val_cv.tolist())
    y_pred_all_loso.extend(y_pred.tolist())
    y_score_all_loso.append(y_pred_proba)

    # Métriques par fold (par sujet)
    loss_cv, acc_cv = model_cv.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
    f1_macro = f1_score(y_val_cv, y_pred, average='macro', zero_division=0)

    print(f"Fold {fold_no} - Accuracy: {acc_cv:.4f} | F1-macro: {f1_macro:.4f}")
    fold_results_loso['accuracy'].append(acc_cv)
    fold_results_loso['f1_macro'].append(f1_macro)
    fold_no += 1

# Empiler les probabilités en un seul tableau (N_samples, n_classes)
if y_score_all_loso:
    y_score_all_loso = np.vstack(y_score_all_loso)
else:
    y_score_all_loso = np.empty((0, num_classes))

end_time_loso = time.time()
print(f"\n--- Résultats de la Validation LOSO ---")
print(f"Précision Moyenne (Accuracy): {np.mean(fold_results_loso['accuracy']):.4f}")
print(f"F1-Macro Moyen : {np.mean(fold_results_loso['f1_macro']):.4f}")
print(f"Temps total LOSO: {end_time_loso - start_time_loso:.2f} secondes")


# %% ------------------------------------------------------------------
# 7. ENTRAÎNEMENT FINAL ET ÉVALUATION SUR LE TEST SET
# --------------------------------------------------------------------
print("\n--- Entraînement Final sur 80% des données ---")

# 1. Division Train/Test (80% / 20%)
# Le 'test_set' est l'ensemble d'évaluation final (sujets mixtes).
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 2. Mise à l'échelle (fit sur train, transform sur train et test)
scaler_final = config['scaler_class']()
X_train_full_scaled = scaler_final.fit_transform(X_train_full)
X_test_scaled = scaler_final.transform(X_test)
print(f"Taille du set d'entraînement final: {X_train_full_scaled.shape}")
print(f"Taille du set de test final: {X_test_scaled.shape}")

# 3. Créer le modèle final
final_model = create_model()

# 4. Entraîner le modèle final
print("Début de l'entraînement final...")
start_time_final = time.time()
history_final = final_model.fit(
    X_train_full_scaled,
    y_train_full,
    epochs=60, 
    batch_size=config['batch_size'], # Paramètre de la config
    # 10% des données d'entraînement sont utilisés pour piloter l'EarlyStopping
    validation_split=0.1, 
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)
end_time_final = time.time()
print(f"Temps total Entraînement Final: {end_time_final - start_time_final:.2f} secondes")

# %% ------------------------------------------------------------------
# 8. ÉVALUATION FINALE (sur Test Set)
# --------------------------------------------------------------------
print("\n--- Évaluation Finale sur le Test Set (Données Inconnues) ---")

loss_test, accuracy_test = final_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Précision (Accuracy) Finale sur le Test Set: {accuracy_test:.4f}")

# Prédictions (Probabilités et Classes)
y_pred_probs_test = final_model.predict(X_test_scaled)
y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1) 

# --- 8b. Rapport de Classification ---
print("\nRapport de Classification Final sur le Test Set:\n")
target_names = [f'Activity {i+1}' for i in range(num_classes)]
print(classification_report(y_test, y_pred_classes_test, target_names=target_names))

# --- 8c. Scores AUC ---
print("\n--- Calcul des Scores AUC (One-vs-Rest) ---")
# Binariser les labels (format One-Hot) pour le calcul AUC multi-classe
y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
y_pred_scores = y_pred_probs_test 
auc_macro = roc_auc_score(y_test_bin, y_pred_scores, average="macro", multi_class="ovr")
auc_weighted = roc_auc_score(y_test_bin, y_pred_scores, average="weighted", multi_class="ovr")
print(f"Score AUC (Moyenne Macro): {auc_macro:.4f}")
print(f"Score AUC (Moyenne Pondérée): {auc_weighted:.4f}")


# %% ------------------------------------------------------------------
# 9. FONCTIONS DE VISUALISATION (Helpers)
# --------------------------------------------------------------------
# Ces fonctions sont appelées dans la section 10 pour générer les sorties graphiques.

def plot_training_history(history, filename=None):
    """Affiche les courbes d'apprentissage (perte et précision)."""
    h = history.history
    if not h:
        print("Historique d'entraînement vide, graphique ignoré.")
        return
    plt.figure(figsize=(12, 5))
    # Graphique de Précision
    plt.subplot(1, 2, 1)
    plt.plot(h.get('accuracy'), label='Précision (Entraînement)')
    plt.plot(h.get('val_accuracy'), label='Précision (Validation)')
    plt.title('Courbe de Précision (Accuracy)')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True)
    # Graphique de Perte
    plt.subplot(1, 2, 2)
    plt.plot(h.get('loss'), label='Perte (Entraînement)')
    plt.plot(h.get('val_loss'), label='Perte (Validation)')
    plt.title('Courbe de Perte (Loss)')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename: 
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Graphique d'entraînement sauvegardé : {filename}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Matrice de confusion', filename=None):
    """Affiche une matrice de confusion heatmap."""
    if labels is None:
        labels = [f"A{i+1}" for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes (Labels)')
    if filename: 
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardée : {filename}")
    plt.show()

def plot_roc_pr_multi(y_true, y_score, n_classes, filename_prefix=None):
    """Affiche les courbes ROC (Globales) et ROC/PR (par classe)."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # --- Calcul des courbes ROC "Micro" et "Macro" ---
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    auc_val_micro = auc(fpr_micro, tpr_micro)

    all_fpr = []
    all_tpr = []
    for i in range(n_classes):
        fpr_class, tpr_class, _ = roc_curve(y_bin[:, i], y_score[:, i])
        all_fpr.append(fpr_class)
        all_tpr.append(tpr_class)
    
    all_fpr_unique = np.unique(np.concatenate([all_fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr_unique)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr_unique, all_fpr[i], all_tpr[i])
    
    mean_tpr /= num_classes
    fpr_macro = all_fpr_unique
    tpr_macro = mean_tpr
    auc_val_macro = auc(fpr_macro, tpr_macro) 

    # --- Tracer les courbes ROC (Globales) ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_micro, tpr_micro,
             label=f'ROC Micro-moyennée (AUC = {auc_val_micro:0.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr_macro, tpr_macro,
             label=f'ROC Macro-moyennée (AUC = {auc_val_macro:0.3f})',
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbes ROC Multi-classe (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    if filename_prefix:
        fn = f'{filename_prefix}_roc_global.png'
        plt.savefig(fn, dpi=150, bbox_inches='tight')
        print(f"Graphique ROC Global sauvegardé : {fn}")
    plt.show()

    # --- Tracer ROC et PR par Classe ---
    plt.figure(figsize=(14, 6))
    # ROC par classe
    plt.subplot(1, 2, 1)
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
            plt.plot(fpr, tpr, lw=1, label=f'classe {i} (AUC={auc(fpr,tpr):.2f})')
        except Exception:
            pass
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC par classe'); plt.legend(fontsize=8)
    
    # PR par classe
    plt.subplot(1, 2, 2)
    for i in range(n_classes):
        try:
            prec, rec, _ = precision_recall_curve(y_bin[:,i], y_score[:,i])
            plt.plot(rec, prec, lw=1, label=f'classe {i}')
        except Exception:
            pass
    plt.xlabel('Rappel (Recall)'); plt.ylabel('Précision'); plt.title('Précision-Rappel par classe'); plt.legend(fontsize=8)
    
    if filename_prefix:
        fn = f'{filename_prefix}_roc_pr_per_class.png'
        plt.savefig(fn, dpi=150, bbox_inches='tight')
        print(f"Graphique ROC/PR par classe sauvegardé : {fn}")
    plt.show()

def plot_fold_metrics(fold_results, title_prefix='', filename=None):
    """Affiche la performance (Accuracy, F1) pour chaque fold LOSO."""
    if not fold_results['accuracy']:
        print("Aucun résultat de fold LOSO à afficher.")
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fold_results['accuracy'], 'o-', label='accuracy')
    plt.axhline(np.mean(fold_results['accuracy']), color='r', linestyle='--')
    plt.title(f'{title_prefix} Accuracy par fold (Moy: {np.mean(fold_results["accuracy"]):.3f})')
    plt.xlabel('Fold (Sujet)'); plt.grid(True); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(fold_results['f1_macro'], 'o-', label='f1_macro')
    plt.axhline(np.mean(fold_results['f1_macro']), color='r', linestyle='--')
    plt.title(f'{title_prefix} F1 macro par fold (Moy: {np.mean(fold_results["f1_macro"]):.3f})')
    plt.xlabel('Fold (Sujet)'); plt.grid(True); plt.legend()
    
    if filename: 
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Graphique des folds sauvegardé : {filename}")
    plt.show()

# %% ------------------------------------------------------------------
# 10. EXÉCUTION DES VISUALISATIONS
# --------------------------------------------------------------------
print(f"\n--- Génération des visualisations pour le scénario : {TEST_SCENARIO} ---")
# Crée un suffixe unique pour tous les fichiers de sortie de ce scénario
suffix = f"_{TEST_SCENARIO.lower()}" 

# 1) Courbes d'entraînement (final model)
plot_training_history(history_final, f'training_history{suffix}.png')

# 2) Matrice de confusion & rapport sur le Test Set final (Stratifié)
plot_confusion_matrix(y_test, y_pred_classes_test, 
                      title=f'Matrice de Confusion (Test Set, Acc: {accuracy_test:.3f})', 
                      filename=f'cm_test{suffix}.png')

# 3) ROC / Precision-Recall sur le Test Set final (Stratifié)
plot_roc_pr_multi(y_test, y_pred_probs_test, num_classes, filename_prefix=f'test_eval{suffix}')

# 4) Matrice de confusion et métriques par fold (LOSO)
if len(y_val_all_loso) > 0:
    # Matrice de confusion agrégée de TOUS les folds LOSO
    plot_confusion_matrix(y_val_all_loso, y_pred_all_loso, 
                          title=f'Matrice de Confusion Agrégée (LOSO, Acc: {np.mean(fold_results_loso["accuracy"]):.3f})', 
                          filename=f'cm_loso{suffix}.png')
    # Performance par fold (sujet)
    plot_fold_metrics(fold_results_loso, title_prefix='LOSO', filename=f'folds_loso{suffix}.png')

# 5) Sauvegarde du modèle final et du scaler
final_model.save(f'final_model{suffix}.h5')
joblib.dump(scaler_final, f'scaler_final{suffix}.pkl')
print(f"Modèle et scaler sauvegardés avec le suffixe : {suffix}")

print("Fin du script.")