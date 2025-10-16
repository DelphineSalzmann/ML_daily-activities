import numpy as np
import os

#Fonction qui renvoie l'ensemble des données correspondant à une activité pour 1 personne

def a_pTab(folder_path):

    folder_path=folder_path

    # Liste pour stocker les tableaux
    all_data = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data = np.loadtxt(file_path, delimiter=',')
            all_data.append(data)

    # Fusion en 1 unique tableau
    total_data = np.vstack(all_data)

    return total_data

#Création d'un dictionnaire par activité avec en clef la personne  et en valeur le tableau des mesures des capteurs
#Autrement dit, on rassemble par label les échantillons en faisant correspondre à chaque échantillon ses features

a01={}
for i in range (1,9):
    folder_path=f'data/a01/p{i}'
    a01[f"p{i}"] = a_pTab(folder_path)

a02={}
for i in range (1,9):
    folder_path=f'data/a02/p{i}'
    a02[f"p{i}"] = a_pTab(folder_path)

a03={}
for i in range (1,9):
    folder_path=f'data/a03/p{i}'
    a03[f"p{i}"] = a_pTab(folder_path)

# a04={}
# for i in range (1,9):
#     folder_path=f'data/a04/p{i}'
#     a04[f"p{i}"] = a_pTab(folder_path)

# a05={}
# for i in range (1,9):
#     folder_path=f'data/a05/p{i}'
#     a05[f"p{i}"] = a_pTab(folder_path)

# a06={}
# for i in range (1,9):
#     folder_path=f'data/a06/p{i}'
#     a06[f"p{i}"] = a_pTab(folder_path)

# a07={}
# for i in range (1,9):
#     folder_path=f'data/a07/p{i}'
#     a07[f"p{i}"] = a_pTab(folder_path)

# a08={}
# for i in range (1,9):
#     folder_path=f'data/a08/p{i}'
#     a08[f"p{i}"] = a_pTab(folder_path)

# a09={}
# for i in range (1,9):
#     folder_path=f'data/a09/p{i}'
#     a09[f"p{i}"] = a_pTab(folder_path)

# a10={}
# for i in range (1,9):
#     folder_path=f'data/a10/p{i}'
#     a10[f"p{i}"] = a_pTab(folder_path)

# a11={}
# for i in range (1,9):
#     folder_path=f'data/a11/p{i}'
#     a11[f"p{i}"] = a_pTab(folder_path)

# a12={}
# for i in range (1,9):
#     folder_path=f'data/a12/p{i}'
#     a12[f"p{i}"] = a_pTab(folder_path)

# a13={}
# for i in range (1,9):
#     folder_path=f'data/a13/p{i}'
#     a13[f"p{i}"] = a_pTab(folder_path)

# a14={}
# for i in range (1,9):
#     folder_path=f'data/a14/p{i}'
#     a14[f"p{i}"] = a_pTab(folder_path)

a15={}
for i in range (1,9):
    folder_path=f'data/a15/p{i}'
    a15[f"p{i}"] = a_pTab(folder_path)

# a16={}
# for i in range (1,9):
#     folder_path=f'data/a16/p{i}'
#     a16[f"p{i}"] = a_pTab(folder_path)

# a17={}
# for i in range (1,9):
#     folder_path=f'data/a17/p{i}'
#     a17[f"p{i}"] = a_pTab(folder_path)

# a18={}
# for i in range (1,9):
#     folder_path=f'data/a18/p{i}'
#     a18[f"p{i}"] = a_pTab(folder_path)

# a19={}
# for i in range (1,9):
#     folder_path=f'data/a19/p{i}'
#     a19[f"p{i}"] = a_pTab(folder_path)