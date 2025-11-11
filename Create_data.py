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

# --- 1. Chargement et Préparation des Données ---

def load_segments_for_activity_person(base_path, activity_id, person_id):
    """
    Charge les 60 segments pour une activité et une personne données.
    Gère les fichiers manquants ou corrompus.
    """
    folder_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}")
    
    if not os.path.exists(folder_path):
        # Avertit si un dossier (ex: data/a04/p1) n'existe pas
        print(f"AVERTISSEMENT: Dossier non trouvé, ignoré : {folder_path}")
        return None
        
    all_segments_data = []
    segments_loaded = 0
    
    for i in range(1, NUM_SEGMENTS_PER_SUBJECT + 1):
        filename = f"s{i:02d}.txt"
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(file_path):
            # Ignore silencieusement si un segment (ex: s60.txt) est manquant
            continue
            
        try:
            # Charger le segment depuis le fichier texte
            segment_data = np.loadtxt(file_path, delimiter=',')
            
            # Vérification critique de la forme (shape) des données
            if segment_data.shape == (SAMPLES_PER_SEGMENT, NUM_SENSORS):
                all_segments_data.append(segment_data)
                segments_loaded += 1
            else:
                # Si la forme est incorrecte, on le signale et on ne charge pas
                print(f"ERREUR: Forme incorrecte dans {filename}. Attendu {(SAMPLES_PER_SEGMENT, NUM_SENSORS)}, Reçu {segment_data.shape}")
        
        except Exception as e:
            # Gérer les fichiers texte vides ou mal formatés
            print(f"ERREUR: Impossible de charger {filename}. Détail : {str(e)}")
    
    # print(f"Chargé {segments_loaded} segments depuis {folder_path}") # Optionnel
    return all_segments_data if all_segments_data else None

def extract_features(segment_data):
    """
    Transforme un segment temporel (125, 45) en un vecteur de caractéristiques (180,).
    Nous utilisons des statistiques simples (moyenne, écart-type, min, max) 
    pour chaque capteur. C'est l'étape d'Ingénierie des Caractéristiques (Feature Engineering).
    """
    features = []
    # Boucle sur chacun des 45 capteurs (colonnes)
    for sensor_col in range(NUM_SENSORS):
        sensor_signal = segment_data[:, sensor_col]
        
        # Calcul des 4 caractéristiques statistiques
        features.append(np.mean(sensor_signal))
        features.append(np.std(sensor_signal))
        features.append(np.min(sensor_signal))
        features.append(np.max(sensor_signal))
        
    # Retourne un seul vecteur de 45 * 4 = 180 caractéristiques
    return np.array(features)

def load_all_data(base_path, num_activities, num_subjects):
    """
    Orchestre le chargement de tous les fichiers, l'extraction de caractéristiques
    et l'imputation des données manquantes.
    """
    all_features = []
    all_labels = []
    print("Démarrage du chargement des données et de l'extraction de caractéristiques...")
    
    # Boucle sur les 19 dossiers d'activités
    for activity_label in range(num_activities):
        activity_id = activity_label + 1
        # print(f"Traitement Activité a{activity_id:02d}...") # Verbeux
        
        # Boucle sur les 8 dossiers de sujets
        for person_id in range(1, num_subjects + 1):
            segments = load_segments_for_activity_person(base_path, activity_id, person_id)
            
            # Si des segments ont été trouvés et chargés pour ce sujet
            if segments:
                for segment in segments:
                    # Conversion du segment (125, 45) en features (180,)
                    features = extract_features(segment)
                    all_features.append(features)
                    # Ajout de l'étiquette (de 0 à 18)
                    all_labels.append(activity_label)

    print(f"Chargement terminé. {len(all_features)} segments trouvés.")
    
    # Vérification de sécurité
    if not all_features:
        raise ValueError("Aucune caractéristique n'a été extraite. Vérifiez les chemins et la configuration (SAMPLES_PER_SEGMENT).")

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"Shape initiale de X: {X.shape}") # (9120, 180)
    print(f"Shape initiale de y: {y.shape}") # (9120,)

    # --- Gestion des Données Manquantes (NaN) ---
    # Cette étape est cruciale si certains fichiers étaient corrompus
    print("Gestion des valeurs manquantes (imputation)...")
    nan_count = np.isnan(X).sum()
    print(f"Nombre de NaN avant imputation: {nan_count}")

    if nan_count > 0:
        # Remplacer tous les NaN par la moyenne de leur colonne (feature)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print(f"Nombre de NaN après imputation: {np.isnan(X).sum()}")
    else:
        print("Aucune valeur NaN détectée.")

    return X, y