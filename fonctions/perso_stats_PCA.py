
#import bibliothéque python

#bibliotheque objet

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#bibliotheque graphique

import seaborn as sns
import matplotlib.pyplot as plt

#inspirer fortement du cours openclassroom : https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees

def analyse_vecteurs(pca,nb_composantPCA,nom_colonne):
    """
    Affiche l'éboulie pour la PCA
    parameter :
    pca : objet pca fiter avec les données
    nom_colonne : nom des colonnes des parametres
    nb_comnposantPCA : nombre de composantes principales  
    
    """

    # on multiplie par 100 et arrondie pour avoir des jolies pourcentages
    eboulie = (pca.explained_variance_ratio_*100).round(2)
    # on fait la somme cumulé
    eboulie_cumul = eboulie.cumsum().round()

    #création d une liste pour afficher l'éboulie (fait bugguer avec le choix du niveau de variance)
    x_list = range(1,nb_composantPCA+1)
    list(x_list)
    

    #affichage eboulie

    plt.bar(x_list, eboulie)
    plt.plot(x_list, eboulie_cumul,c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    #preparation heatmap
    pcs = pca.components_
    pcs = pd.DataFrame(pcs)
    pcs.columns = nom_colonne
    pcs.index = [f"F{i}" for i in x_list]

    #affichage heatmap
    fig, ax = plt.subplots(figsize=(20, 6))

    sns.heatmap(pcs.T, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")
    plt.show()


def centrer_reduit(X):

    """src cours openclassroom
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees/5280368-comprenez-lenjeu-de-lanalyse-en-composantes-principales
    fait un centrer reduit en vue d une PCA
    paramater :
        df : dataframe sur lequel faire la PCA
    return :
        X_scaled : dataframe centré reduit
        names : noms en index
        nom_colonne : nom des colonnes
    
    """
    
    # On instancie notre scaler : 
    scaler = StandardScaler()
    
    names = X.index

    nom_colonne = X.columns

    #on applique le standart scaler : (operation fit + transform)
    X_scaled = scaler.fit_transform(X)


    # On le transforme en DataFrame : 
    X_scaled = pd.DataFrame(X_scaled)

    # On peut appliquer la méthode .describe() et .round()
    display(X_scaled.describe().round(2))

    #On vois que l'on est centrée (moyenne a 0 et reduit ecart-type a 1)
    return(X_scaled,names,nom_colonne)

def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)

def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

     
    # Les points
    if clusters == None :
        plt.scatter(   X_[:, x], 
                        X_[:, y], 
                        alpha=alpha, 
                        marker=marker)
    else:    
        plt.scatter(   X_[:, x], 
                            X_[:, y], 
                            alpha=alpha, 
                            c=clusters, 
                            cmap="Set1", 
                            marker=marker)


    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()

def ACP(X,nbcomposantePCA):
    #centrer et reduire les variables
    X_scaled,names,nom_colonne = centrer_reduit(X)

    #eboulie et heatmap (sur toute les données)
    pca = PCA(n_components=X_scaled.shape[1])
    pca.fit(X_scaled)
    analyse_vecteurs(pca,X_scaled.shape[1],nom_colonne)
    
    #selection du nombre de composante : 
    nb_composantPCA=0
    if nbcomposantePCA <=0 :
        nb_composantPCA = X_scaled.shape[1]
    else :
        nb_composantPCA = nbcomposantePCA

    #selectionne la variance voulu (mais fait bugguer la suite du programme. Je garde cette fonction a but pedagogique)
    #nb_composantPCA = 0.9 

    #credeinir le pca pour le nombre de composantes desirées

    pca = PCA(n_components=nb_composantPCA)
    
    #Fiter le PCA
    pca.fit(X_scaled)
    
    #verifier si on a garder la variance necessaire
    print("Total de la variance captée par la PCA :")
    print(pca.explained_variance_ratio_.sum()*100,"%")
    
 
    
    #affichage des correlations  F1 F2
    correlation_graph(pca, (0,1), nom_colonne)
    
    #projection dans la dimension de la PCA
    X_proj = pca.transform(X_scaled)
    
    #affichage projection
    display_factorial_planes(X_proj, (0,1), pca, labels=names, figsize=(20,16), marker="o")
    
    return pca,X_scaled,X_proj,names,nom_colonne