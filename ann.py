# %% Importations nécessaires
# Bibliothèques standards
import os
import numpy as np

# Bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Bibliothèques TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers
import keras_tuner as kt

# Vérification des versions
print(f"TensorFlow Version: {tf.__version__}")
print(f"KerasTuner Version: {kt.__version__}")

# --- Configuration ---
NUM_ACTIVITIES = 19
NUM_SUBJECTS = 8
NUM_SEGMENTS_PER_SUBJECT = 60
NUM_SENSORS = 45
SAMPLES_PER_SEGMENT = 125
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), 'data') # Assurez-vous que c'est le bon chemin
N_SPLITS_CV = 5 # Nombre de folds pour la validation croisée
RANDOM_STATE = 42 # Pour la reproductibilité

# --- 1. Chargement et Préparation des Données ---

def load_segments_for_activity_person(base_path, activity_id, person_id):
    folder_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}")
    print(f"Checking path: {folder_path}")
    
    # Vérification détaillée
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return None
        
    # Liste les fichiers trouvés
    files = os.listdir(folder_path)
    print(f"Found {len(files)} files in {folder_path}")
    
    all_segments_data = []
    segments_loaded = 0
    
    for i in range(1, NUM_SEGMENTS_PER_SUBJECT + 1):
        filename = f"s{i:02d}.txt"
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(file_path):
            print(f"Missing file: {filename}")
            continue
            
        try:
            segment_data = np.loadtxt(file_path, delimiter=',')
            if segment_data.shape == (SAMPLES_PER_SEGMENT, NUM_SENSORS):
                all_segments_data.append(segment_data)
                segments_loaded += 1
            else:
                print(f"Wrong shape in {filename}: {segment_data.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    print(f"Successfully loaded {segments_loaded} segments from {folder_path}")
    return all_segments_data if all_segments_data else None

def extract_features(segment_data):
    """ Extrait moyenne, std, min, max pour chaque capteur. Gère les NaN. """
    features = []
    if segment_data is None: # Si le segment entier est manquant
        return np.full(NUM_SENSORS * 4, np.nan)

    for sensor_col in range(NUM_SENSORS):
        sensor_signal = segment_data[:, sensor_col]
        # Ignore les NaN dans les calculs si présents
        features.append(np.nanmean(sensor_signal))
        features.append(np.nanstd(sensor_signal))
        features.append(np.nanmin(sensor_signal))
        features.append(np.nanmax(sensor_signal))
    return np.array(features)

def load_all_data(base_path, num_activities, num_subjects):
    """ Charge toutes les données, extrait les features et retourne X, y """
    all_features = []
    all_labels = []
    print("Starting data loading and feature extraction...")
    for activity_label in range(num_activities):
        activity_id = activity_label + 1
        print(f"Processing Activity a{activity_id:02d}...")
        for person_id in range(1, num_subjects + 1):
            segments = load_segments_for_activity_person(base_path, activity_id, person_id)
            if segments:
                for segment in segments:
                    features = extract_features(segment)
                    all_features.append(features)
                    all_labels.append(activity_label)

    print(f"Data loading finished. Found {len(all_features)} segments.")
    if not all_features:
        raise ValueError("No features were extracted. Check data paths and file contents.")

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"Initial shape of X: {X.shape}")
    print(f"Initial shape of y: {y.shape}")

    # --- Gestion des NaN ---
    print("Handling potential missing values (imputation)...")
    # Vérifier s'il y a des NaN
    nan_count = np.isnan(X).sum()
    print(f"Number of NaN values before imputation: {nan_count}")

    if nan_count > 0:
        # Remplacer les NaN par la moyenne de la colonne (feature)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print(f"Number of NaN values after imputation: {np.isnan(X).sum()}")
    else:
        print("No NaN values found.")

    return X, y

# --- Charger les données ---
X_raw, y = load_all_data(DATA_BASE_PATH, NUM_ACTIVITIES, NUM_SUBJECTS)
num_features = X_raw.shape[1]
num_classes = NUM_ACTIVITIES

# --- 2. Construction du Modèle ANN (Fonction pour Keras Tuner et CV) ---

def build_model(hp):
    """ Fonction pour construire le modèle Keras, paramétrable par Keras Tuner. """
    model = keras.Sequential(name="mlp_activity_classifier")
    model.add(layers.Input(shape=(num_features,), name="input_layer"))

    # Hyperparamètre : Nombre d'unités dans la première couche dense
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(layers.Dense(units=hp_units_1, activation='relu', name='hidden_layer_1'))

    # Hyperparamètre : Taux de Dropout 1
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout_1))

    # Hyperparamètre : Nombre d'unités dans la deuxième couche dense
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    model.add(layers.Dense(units=hp_units_2, activation='relu', name='hidden_layer_2'))

    # Hyperparamètre : Taux de Dropout 2
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout_2))

    model.add(layers.Dense(num_classes, activation='softmax', name='output_layer'))

    # Hyperparamètre : Taux d'apprentissage
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 3. Recherche d'Hyperparamètres avec Keras Tuner ---
print("\n--- Starting Hyperparameter Tuning ---")

# Diviser une petite partie pour le tuning (pour aller plus vite)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train_tune, _, y_train_tune, _ = train_test_split(
    X_train_full, y_train_full, train_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full # Utiliser 25% de 80% = 20% du total
)

# Mise à l'échelle spécifique pour le tuning set
scaler_tune = StandardScaler()
X_train_tune_scaled = scaler_tune.fit_transform(X_train_tune)

# Configurer le tuner (RandomSearch est rapide pour commencer)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Nombre de combinaisons d'hyperparamètres à essayer
    executions_per_trial=1, # Pour la robustesse, on pourrait faire > 1
    directory='keras_tuner_dir',
    project_name='activity_classification_tuning'
)

# Lancer la recherche
# Utiliser un callback pour arrêter tôt si la validation loss ne s'améliore plus
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train_tune_scaled, y_train_tune,
             epochs=30, # Moins d'époques pour le tuning
             validation_split=0.2, # Validation sur une partie du tuning set
             callbacks=[stop_early],
             verbose=1)

# Obtenir les meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Hyperparameter search complete.
Best number of units in layer 1: {best_hps.get('units_1')}
Best dropout rate for layer 1: {best_hps.get('dropout_1')}
Best number of units in layer 2: {best_hps.get('units_2')}
Best dropout rate for layer 2: {best_hps.get('dropout_2')}
Best learning rate: {best_hps.get('learning_rate')}
""")

# Construire le modèle final avec les meilleurs hyperparamètres trouvés
best_model = build_model(best_hps)

# --- 4. Validation Croisée (Cross-Validation) ---
print("\n--- Starting Cross-Validation with Best Hyperparameters ---")

# Utiliser StratifiedKFold pour conserver la proportion des classes dans chaque fold
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

fold_accuracies = []
fold_histories = [] # Pour stocker l'historique de chaque fold

fold_no = 1
# Utiliser les données complètes (X_raw, y) pour la CV
for train_index, val_index in skf.split(X_raw, y):
    print(f"\n--- Fold {fold_no}/{N_SPLITS_CV} ---")
    X_train_cv, X_val_cv = X_raw[train_index], X_raw[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]

    # Mise à l'échelle DANS la boucle de CV
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv) # transform seulement sur la validation

    # Construire un nouveau modèle pour chaque fold (important!)
    # Il est préférable de re-construire et re-compiler pour réinitialiser les poids
    model_cv = build_model(best_hps) # Utilise les meilleurs HPs trouvés

    # Entraîner le modèle
    print(f"Training Fold {fold_no}...")
    history_cv = model_cv.fit(
        X_train_cv_scaled,
        y_train_cv,
        epochs=50, # Nombre d'époques pour l'entraînement final par fold
        batch_size=64,
        validation_data=(X_val_cv_scaled, y_val_cv), # Utiliser le set de validation du fold
        verbose=0 # Moins verbeux dans la boucle
    )
    fold_histories.append(history_cv)

    # Évaluer sur le set de validation du fold
    loss_cv, accuracy_cv = model_cv.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
    print(f"Fold {fold_no} Validation Accuracy: {accuracy_cv:.4f}")
    fold_accuracies.append(accuracy_cv)

    fold_no += 1

# Afficher les résultats de la validation croisée
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
print("\n--- Cross-Validation Results ---")
print(f"Individual Fold Accuracies: {fold_accuracies}")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")

# Optionnel: Visualiser les courbes d'apprentissage moyennes sur les folds
# (Cela peut être un peu complexe à moyenner proprement)

# --- 5. Entraînement Final et Évaluation sur le Test Set ---
# Optionnellement, on pourrait ré-entraîner le meilleur modèle sur TOUT X_train_full
print("\n--- Final Training on Full Training Set ---")
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full)
X_test_scaled = scaler_final.transform(X_test) # Re-scaler le test set avec le scaler final

final_model = build_model(best_hps)
history_final = final_model.fit(
    X_train_full_scaled,
    y_train_full,
    epochs=60, # Entraîner un peu plus longtemps sur l'ensemble complet
    batch_size=64,
    validation_split=0.1, # Petite validation pour surveiller
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], # Arrêt anticipé
    verbose=1
)

print("\n--- Final Evaluation on Test Set ---")
loss_test, accuracy_test = final_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Final Test Accuracy: {accuracy_test:.4f}")

y_pred_probs_test = final_model.predict(X_test_scaled)
y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1)

# Matrice de Confusion Finale
cm_test = confusion_matrix(y_test, y_pred_classes_test)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,num_classes+1), yticklabels=range(1,num_classes+1))
plt.title('Matrice de Confusion Finale sur le Test Set')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.show()

# Rapport de Classification Final
print("\nRapport de Classification Final sur le Test Set:\n")
target_names = [f'Activity {i+1}' for i in range(num_classes)]
print(classification_report(y_test, y_pred_classes_test, target_names=target_names))