##  Guide d'utilisation

Le code présente deux modèles de prédiction d'activités quotidiennes à partir de données issus de capteurs sur 8 sujets (voir la section suivante pour la description de la base de données) : un modèle de réseau de neurones artificiels (ANN) et un modèle de forêt aléatoire (Random Forest).
Pour que le script fonctionne, le dossier data doit être placé dans le même répertoire que les scripts ann_report.py, mainRandomForest.py, RandomForest_report.py et Get_data.py.

    # ANN

    Le script Python ann_report.py entraîne et évalue le modèle de Réseau de Neurones Artificiels (ANN) pour la classification des 19 activités humaines. Il est conçu pour être configurable, permettant de reproduire tous les tests de l'analyse de sensibilité présentés dans ce rapport.

    Toute la configuration des tests se gère via la variable TEST_SCENARIO en Section 2.
        Le script est configuré par défaut pour exécuter le modèle final optimisé (TEST_SCENARIO = "CHAMPION").

        Pour reproduire les autres tests présentés dans le rapport, modifiez simplement la variable TEST_SCENARIO  (ligne 40) par l'une des valeurs suivantes et relancez le script :
            "BASELINE": Reproduit le modèle initial (Dropout 0.1/0.4, Batch Size 64) .
            "TEST_NO_DROPOUT": Reproduit le test sans aucune régularisation Dropout .
            "TEST_MINMAX_SCALER": Reproduit le test en remplaçant le StandardScaler par un MinMaxScaler .
            "TEST_SIMPLE_ARCHI": Reproduit le test avec l'architecture simplifiée 64x32 .
            "TEST_90_FEATURES": Reproduit le test utilisant seulement 90 features (mean, std).
            Note : Ce test créera un fichier cache séparé (processed_data_90features.npz) .

    Pour chaque scénario exécuté, le script va :

        Afficher dans la console : Les métriques clés, incluant la précision moyenne de la CV Stratifiée , la précision moyenne de la CV LOSO (généralisation) , et la précision finale sur le Test Set.

        Sauvegarder les graphiques : Tous les graphiques (Courbes d'apprentissage, Matrices de confusion, Courbes ROC) sont sauvegardés en .png dans le dossier.

        Suffixe de Fichiers : Tous les fichiers de sortie (graphiques, modèle .h5, scaler .pkl) sont suffixés avec le nom du scénario (ex: cm_loso_champion.png, final_model_champion.h5).


    # Random Forest

    Le fichier main_RandomForest.py est conçu pour être configurable et permet d'intéragir avec le modèle et choisir les sorties souhaitées, permettant de reproduire tous les tests de l'analyse de sensibilité présentés dans ce rapport. Il fait appel aux fonctions écrites dans le fichier RandomForest.py. 
    Il se décompose en 4 parties indépendantes. Dans chaque partie, les éléments d'entrée sont modifiables (#Modifiable). L'appel aux fonctions ne doit pas être modifié, mais il est possible de les commenter si on ne souhaite pas afficher les résultats ou si on veut gagner du temps de calcul. 

        - Une partie pour obtenir les résultats de la méthode Gridsearch pour trouver les meilleurs paramètres parmi une grille. 

        - Une partie pour obtenir les scores de prédiction par validation croisée LOSO ainsi que la matrice de confusion et les courbes ROC et Precision-Recall. Il est possible d'afficher également les scores et la matrice de confusion pour une validation croisée 10-folds. Le modèle peut être modifié. Par défaut, il s'agit du résultat issu de Gridsearch avec les meilleurs paramètres.

        - Une partie pour étudier l'influence des paramètres et obtenir les courbes d'accuracy pour les jeux d'entraînement et de test ainsi que l'impureté de Gini moyenne de la forêt en fonction de la valeur du paramètre étudié. Par défaut, la méthode LOSO est utilisée avec tous les sujets, mais il est possible de ne choisir que quelques sujets, notammetn pour réduire les temps de calcul.

        - Une partie pour afficher l'impureté de Gini moyenne par arbre pour un unique entraînement sur 7 des 8 sujets et test sur le 8ème sujet.

    Il suffit ensuite d'exécuter le fichier. Pour chaque exécution, le script va :
        Afficher dans la console les métriques clés demandées
        Afficher les graphes demandés

# Prérequis 

Python 3.8

Bibliothèques :
    numpy
    matplotlib
    scikit-learn
    keras
    os
    seaborn
    time
    pandas
    collections
    joblib
    tensorflow
Pour installer toutes les bibiothèques: pip install tensorflow numpy pandas matplotlib seaborn scikit-learn joblib os time collections


# Data base description
The data are from a public database : https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities


Each of the 19 activities is performed by eight subjects (4 female, 4 male, between the ages 20 and 30) for 5 minutes.

Total signal duration is 5 minutes for each activity of each subject.

The subjects are asked to perform the activities in their own style and were not restricted on how the activities should be performed. For this reason, there are inter-subject variations in the speeds and amplitudes of some activities. 

The activities are performed at the Bilkent University Sports Hall, in the Electrical and Electronics Engineering Building, 

and in a flat outdoor area on campus. Sensor units are calibrated to acquire data at 25 Hz sampling frequency. The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity.


The 19 activities are: 

sitting (A1), 
standing (A2), 
lying on back and on right side (A3 and A4), 
ascending and descending stairs (A5 and A6), 
standing in an elevator still (A7) 
and moving around in an elevator (A8), 
walking in a parking lot (A9), 
walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11),
running on a treadmill with a speed of 8 km/h (A12), 
exercising on a stepper (A13), 
exercising on a cross trainer (A14), 
cycling on an exercise bike in horizontal and vertical positions (A15 and A16),
rowing (A17), 
jumping (A18), 
and playing basketball (A19).


File structure:

19 activities (a) (in the order given above)

 8 subjects   (p)

60 segments   (s)

 5 units on torso (T), right arm (RA), left arm (LA), right leg (RL), left leg (LL)

 9 sensors on each unit (x,y,z accelerometers, x,y,z gyroscopes, x,y,z magnetometers)
 

Folders a01, a02, ..., a19 contain data recorded from the 19 activities.


For each activity, the subfolders p1, p2, ..., p8 contain data from each of the 8 subjects.


In each subfolder, there are 60 text files s01, s02, ..., s60, one for each segment.


In each text file, there are 5 units x 9 sensors = 45 columns and 5 sec x 25 Hz = 125 rows.

Each column contains the 125 samples of data acquired from one of the sensors of one of the units over a period of 5 sec.

Each row contains data acquired from all of the 45 sensor axes at a particular sampling instant separated by commas.



Columns 1-45 correspond to:  

 T_xacc,  T_yacc,  T_zacc,  T_xgyro, ...,  T_ymag,  T_zmag,

RA_xacc, RA_yacc, RA_zacc, RA_xgyro, ..., RA_ymag, RA_zmag,

LA_xacc, LA_yacc, LA_zacc, LA_xgyro, ..., LA_ymag, LA_zmag,

RL_xacc, RL_yacc, RL_zacc, RL_xgyro, ..., RL_ymag, RL_zmag,

LL_xacc, LL_yacc, LL_zacc, LL_xgyro, ..., LL_ymag, LL_zmag.



Therefore,

columns  1-9  correspond to the sensors in unit 1 (T), 

columns 10-18 correspond to the sensors in unit 2 (RA), 

columns 19-27 correspond to the sensors in unit 3 (LA), 

columns 28-36 correspond to the sensors in unit 4 (RL), 

columns 37-45 correspond to the sensors in unit 5 (LL). 

