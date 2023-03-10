from IPython.display import display
import pandas as pd

def ecris_liste(liste,chemin_fichier,sep=";"):
    """ecris une liste dans un fichier
    Arg :
        Liste : liste a écrire
        sep : separteur utilisé
        chemin_fichier : chemin du fichier + nom du fichier
    """
    fichier = open(chemin_fichier,'w')
    strliste =sep.join([str(elem) for elem in liste ])    

    fichier.write(strliste)
    fichier.close()

def lis_liste(chemin_fichier,sep=';'):
    """
    lit un fichier et retourne une liste
    Arg :
        sep : separteur utilisé
        chemin_fichier : chemin du fichier + nom du fichier
    Return :
        liste : liste lu dans le fichier
    """
    fichier = open(chemin_fichier,'r')
    strliste = (fichier.read())
    liste = strliste.split(sep)
    fichier.close()
    return liste

def detection_clé(df):
    """indique si une colonne peut être utilisé en clé
    Args:
        df : dataframe as DataFrame
    
    Returns :
        None"""

    cle_non_detecte = True
    for x in df:
        if len(df[x].unique()) == df.shape[0]: #si le nombre de ligne est egal au nombre d'élement unique alors c'une clé
            print("la colonne",x,"est une clé")
            cle_non_detecte = False
    if cle_non_detecte:
        print("pas de clé") 

def analyseDataFrame(df):
    """affiche un resumé des Dataframes info describe head et verification doublons

    Args:
        df : dataframe à analyser as dataFrame

    Returns:
        None"""

    print ("################################# INFO  #################################")
    df.info()
    print ("################################# DESCRIBE  #################################")
    display(df.describe(include='all',datetime_is_numeric=True))
    print ("################################# HEAD  #################################")
    display(df.head())
    print ("################################# DOUBLONS  #################################")
    display(df[df.duplicated()])
    print("################################## recherche de clé pour merge ###################")
    detection_clé(df)
    
def categoriser(data,nom_col_recherche,mot_cle,nom_categorie,nom_col_categorie='categorie',regex=False):
    """recherche un mot dans une string et enregistre un nom de categorie dans la colonne categorie (par defaut)
    Args:
        data : dataframe à catégoriser as DataFrame
        nom_col_recherche : colonne où rechercher as string
        nom_categorie : nom de la catégorie as string default : 'categorie
        nom_col_categorie : nom de la colonne cat"egorie a créer as string
        regex : utilisation de regex as bool default: False
    Returns :
        None"""
    data.loc[data[nom_col_recherche].str.contains(mot_cle,regex=regex) , nom_col_categorie] = nom_categorie


if __name__ == '__main__':
    df = pd.DataFrame({'A': [1.1, 2.7, 5.3],
                       'B': [2, 10, 9],
                       'C': [1.0, 5.4, 1.0],
                       'D': [4, 15, 15]},
                      index = ['a1', 'a2', 'a3'])
    lst = [["aliment sandwich",'aliment'],
           ["aliment Jambon","aliment"],
           ['outil clé a molette','aliment'],
           ['outil tournevis','aliment']]
    df2 = pd.DataFrame(lst, columns = ['Nom','type'])
    
    print("######### exemple d'analyseDataFrame ##########")
    analyseDataFrame(df)

    print("######### exemple detection clé dataframe ########")
    detection_clé(df)

    print("######### exemple categorize ########")
    print("##### dataframe d'exemple #######")
    print(df2.head())

    print("categoriser(df2,'Nom','outil','outilage','type')")
    categoriser(df2,'Nom','outil','outilage','type')
    print(df2.head())    
