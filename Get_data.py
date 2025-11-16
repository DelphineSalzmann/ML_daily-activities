import os
import numpy as np

# Bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques scikit-learn

from sklearn.impute import SimpleImputer

# --- Configuration ---
NUM_ACTIVITIES = 19
NUM_SUBJECTS = 8
NUM_SEGMENTS_PER_SUBJECT = 60
NUM_SENSORS = 45
SAMPLES_PER_SEGMENT = 125
config={'features':180}

# --- 1. Chargement et Préparation des Données ---

def load_segments_for_activity_person(base_path, activity_id, person_id):
    # ... (fonction inchangée, copiée de ann_bis_batch32.py) ...
    folder_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}")
    if not os.path.exists(folder_path):
        print(f"AVERTISSEMENT: Dossier non trouvé, ignoré : {folder_path}")
        return None
    all_segments_data = []
    segments_loaded = 0
    for i in range(1, NUM_SEGMENTS_PER_SUBJECT + 1):
        filename = f"s{i:02d}.txt"
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            continue
        try:
            segment_data = np.loadtxt(file_path, delimiter=',')
            if segment_data.shape == (SAMPLES_PER_SEGMENT, NUM_SENSORS):
                all_segments_data.append(segment_data)
                segments_loaded += 1
            else:
                print(f"ERREUR: Forme incorrecte dans {filename}. Attendu {(SAMPLES_PER_SEGMENT, NUM_SENSORS)}, Reçu {segment_data.shape}")
        except Exception as e:
            print(f"ERREUR: Impossible de charger {filename}. Détail : {str(e)}")
    return all_segments_data if all_segments_data else None

def extract_features(segment_data):
    """
    Transforme un segment temporel (125, 45) en un vecteur de caractéristiques.
    Le nombre de features (90 ou 180) dépend de la configuration globale.
    """
    features = []
    
    # Gérer les segments manquants
    if segment_data is None:
        num_stats = 2 if config['features'] == 90 else 4
        return np.full(NUM_SENSORS * num_stats, np.nan)

    # Boucle sur chacun des 45 capteurs (colonnes)
    for sensor_col in range(NUM_SENSORS):
        sensor_signal = segment_data[:, sensor_col]
        
        # Caractéristiques de base (toujours incluses)
        features.append(np.mean(sensor_signal))
        features.append(np.std(sensor_signal))
        
        # N'ajoute min/max QUE si le config le demande
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
    all_groups = []   
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
                        all_groups.append(person_id)

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
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print(f"Nombre de NaN après imputation: {np.isnan(X).sum()}")
    else:
        print("Aucune valeur NaN détectée.")

    return X, y, groups   


