import matplotlib.pyplot as plt
from Create_data import a01, a02, a03, a15


def Visualiser(capteurs, nom_dico, activités,personnes, nom_courbe):
    '''
    capteurs, activités et personnes sont des listes
    nom_dico est la liste des str contenant le nom des activités
    nom_courbe et le titre qu'on souhaite donner à la courbe

    La fonction renvoie le graphe des mesures pour les capteurs, personnes et activités spécifiées
    '''
    plt.figure(figsize=(10, 5))
    i=0
    for a in activités:
        for cle in personnes:
            for capteur in capteurs:
                plt.plot(a[cle][:, capteur], label= f"{nom_dico[i]}{cle} - capteur {capteur+1}")
        i+=1
        

    plt.title(nom_courbe)
    plt.xlabel("Temps")
    plt.ylabel("Valeur mesurée")
    plt.legend()
    plt.grid(True)
    plt.show()


capteurs=[9]
activités=[a01,a02,a15]
personnes=["p1","p3","p7"]
nom_dico=["a01","a02","a15"]
nom_courbe="Comparaison pour 3 activités des données d'un capteur au cours du temps"

Visualiser(capteurs, nom_dico, activités, personnes, nom_courbe)