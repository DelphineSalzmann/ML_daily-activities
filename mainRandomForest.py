from RandomForest_report import best_model, validation_LOSO,Influence_n_estimators_LOSO,Influence_nleaf_LOSO,Influence_max_depth_LOSO, plot_PR,plot_confusion,plot_ROC, calcul_gini, validiation_croisée, CM
from sklearn.ensemble import RandomForestClassifier

############################################################################################
# -- Déterminer les meilleurs paramètres au sein d'une grille par validation croisée LOSO (LeaveOneSubjectOut) --

# Attention, plus la grille est fournie, plus le temps de calcul est important. Comptez au moins une vingtaine de minutes.
param_grid = {                                 #A modifier
        'n_estimators': [100,200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }                       

#Décommenter la ligne ci-dessous pour obtenir les meilleurs paramètres parmi la grille définie
#grid_search = best_model(param_grid)

#############################################################################################
# -- Obtenir la matrice de confusion et les courbes ROC et Precision-recall par méthode LOSO

#Modifiable
rf = RandomForestClassifier(random_state=42, bootstrap= True, max_depth= None, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 200)


#Décommenter les lignes ci-dessous pour  obtenir les résultats de la validation croisée LOSO et afficher la matrice de confusion, les courbes ROC et PR
cm, y_true_all, y_score_all=validation_LOSO(rf)
plot_confusion(cm)
plot_ROC(y_true_all,y_score_all)
plot_PR(y_true_all,y_score_all)


################# Comparaison avec Validation croisée K-folds stratifiés
#validiation_croisée(rf)
#CM(rf)

#############################################################################################
# -- Faire varier les paramètres --

#Les scores sont calculés comme la moyenne des validations croisées LOSO qui excluent successivement les sujets indiqués dans la liste.
#Une liste moins longue implique moins de plis dans la validation croisée mais un temps de calcul plus court
liste_exclus=[1,2,3,4,5,6,7,8]    #Modifiable

#Décommenter les lignes ci-dessous pour afficher les courbes
# Influence_n_estimators_LOSO(liste_exclus)
# Influence_nleaf_LOSO(liste_exclus)
# Influence_max_depth_LOSO(liste_exclus)

###############################################################################################
# -- Impureté de Gini - LeaveOneSubjectOut --

# Affiche l'impureté de chaque arbre et l'impureté moyenne de la forêt avec test sur les données du sujet choisi

sujet = 1        #Au choix parmi 1,2,3,4,5,6,7,8

#calcul_gini(sujet, rf)
