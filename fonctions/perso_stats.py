from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

#### Courbe pour stats ######
def courbe_Lorentz(plt_ax,df,colonne,titre="",titreX="",titreY=""):
    """configure une courbe de Lorentz sur l'axe matplotlib
    Args:
        plt_ax : Axe Matplotlib as matplotlib.axes_subplots.AxesSubplot
        df : dataframe a exploiter as DataFrame
        colonne: nom de la colonne pour faire le graphe as string
        titre : titre du graphe as string default: ""
        titreX : titre de l'axe X as string default: ""
        titreY: titre de l'axe Y as string default: ""
    Returns :
        indice de Gini as float"""

    #classer les valeur en fonction du chiffre d'affaire
    data_lorentz = df.sort_values(by=colonne,ascending=False).copy(deep=True)

    #enlevé les Nan
    data_lorentz = data_lorentz[~data_lorentz[colonne].isna()]

    #création de colonnes pour le graphe de lorentz

    #CA cumulé
    data_lorentz['pourcentage Y cumulé'] = (data_lorentz[colonne]/data_lorentz[colonne].sum()*100).cumsum()

    #pourcentage de bouteille je me sers d'un nouvel index
    data_lorentz = data_lorentz.reset_index()
    
    data_lorentz['pourcentage index']=data_lorentz.index/(data_lorentz.shape[0]-1)*100

    # creation d'une courbe de Lorentz
    #sns.lineplot(data =data_lorentz, x='pourcentage index', y='pourcentage Y cumulé',ax=plt_ax)
    plt_ax.plot(data_lorentz['pourcentage index'],data_lorentz['pourcentage Y cumulé'])

    #courbe de lorentz avec repartition ideale
    plt_ax.plot([100, 0],[100,0])

    #titre,et titre des axes
    plt_ax.set_title(titre)
    plt_ax.set_xlabel(titreX)
    plt_ax.set_ylabel(titreY)    

    # calcul de gini
    surface_bisectrice = 100*100/2
    pas_index = 1/(data_lorentz.shape[0]-1)*100
    data_lorentz['surface_courbe']= data_lorentz['pourcentage Y cumulé']*pas_index
    gini = (data_lorentz['surface_courbe'].sum()-surface_bisectrice)/(5000)
    return gini

############## TEST_STATS ##################


def choix_H0_H1(pvalue,confiance=0.05):
    """ fait un choix entre H0 et H1 en fonction de la confiance et de la pvalue"""
    if pvalue > confiance:
        print("Étant donné que la p-values est supérieure au niveau de signification alpha =",confiance,",\non ne peut pas rejeter l'hypothèse nulle H0.")
        print("Le risque de rejeter l'hypothèse nulle H0 alors qu'elle est vraie est inferieur à ",round(confiance*100,2),"%")
        return True
    else:
        print("Étant donné que la p-values est inférieure au niveau de signification alpha =",confiance,",\non doit rejeter l'hypothèse nulle H0 et retenir l'hypothèse H1.")
        print("Le risque de rejeter l'hypothèse nulle H0 alors qu'elle est vraie est inferieur à ",round(pvalue*100,2),"%")
        return False

#calcul du chi2unem/python-test-du-chi-carre-de-pearson/

#https://sites.google.com/site/testdukhi2/le-khi-2-etape-par-etape lien pour comprendre le khi 2

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html 


def test_chi2 (df,col1,col2,value="pas_de_value",aggfunc='count',confiance=0.05):
    """
    effectue un test du chi 2 sur le dataframe
    Args:
        df : dataframe a exploiter as DataFrame
        col1: nom de la colonne de la premiere variable as string
        col2: nom de la colonne de la deuxieme variable as string
        Value: colonne pour les valeurs aprendre en compte as string default : "pas_de_values"
        aggfunc: fonction aggrementation à utiliser as string default : count
        confiance : indice de confiance as float default : 0.05
    
    Returns :
        result : bool resultat du test
    """

    #préparation du tableau de contingence
    if value == "pas_de_value":
        data_chi = df[[col1,col2]].copy()
        data_chi[value] = 1
    else:
        data_chi = df[[col1,col2,value]].copy()
    
    #data_chi = data_chi.sample(1500)
    data_chi = pd.pivot_table(
                            data=data_chi
                            ,index=col1
                            ,columns=col2
                            ,values=value
                            ,aggfunc=aggfunc
                            )
    #affichage tableau de contingence
    print("tableau de contingence:")
    display(data_chi)

    #test du chi2 par scypy:
    stat , p ,dof ,expected, = stats.chi2_contingency(data_chi)
    print("valeur du chi2",stat)
    print("valeur attendue :\n",expected)
    print("degrés de liberte:",dof)
    print ("valeur de p-values chi2 : ", p)

    print("Interprétation du test:\nH0 : les valeurs sont indépendantes\nH1 : les valeurs sont dépendantes")
    result = choix_H0_H1(p,confiance)
    return result  

#source: https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

#test adf (dectecte s'il y a une racine unitaire)
def adf_test(timeseries,confiance=0.05):
    """ test adf tel que decris dans statsmodels
     Args:
        timeseries : objet pandas        
        confiance : indice de confiance as float default : 0.05
    
    Returns :
        result : bool resultat du test"""

    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    print("\n\nInterprétation du test:\nH0 : La série a une racine unitaire\nH1 : La série n'a pas de racine unitaire")
    result = choix_H0_H1(dftest[1],confiance)
    return result
    
    
#test kpss (teste la stationnarité)

def kpss_test(timeseries,confiance=0.05):
    """ test kpss tel que decris dans statsmodels
     Args:
        timeseries : objet pandas         
        confiance : indice de confiance as float default : 0.05
    
    Returns :
        result : bool resultat du test"""

    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

    print("\n\nInterprétation du test:\nH0 : Le processus est à tendance stationnaire.\nH1 : La série a une racine unitaire (la série n'est pas stationnaire)")
    result = choix_H0_H1(kpsstest[1],confiance)
    return result
    

def test_loi_normale(df,confiance=0.05):
    """ teste si la serie suit une loi normale
    Args:
        df : objet pandas series        
        confiance : indice de confiance as float default : 0.05
    
    Returns :
        result : bool resultat du test"""
    
    stat,pvalue = stats.shapiro(df)
    print("\n\nInterprétation du test:\nH0 : La série suit une loi Normale\nH1 : La série ne suit pas une loi Normale")
    result = choix_H0_H1(pvalue,confiance)
    return result
    

def ANOVA(df1,df2,confiance=0.05):
    """ fait une Anova entre 2 series
    Args:
        df1 : objet pandas series
        df2 : objet pandas series        
        confiance : indice de confiance as float default : 0.05
    
    Returns :
        result : bool resultat du test"""

    stat,pvalue = stats.f_oneway(df1,df2)
    print("\n\nInterprétation du test:\nH0 : Les moyennes des groupes sont égales\nH1 : Les moyennes des groupes ne sont pas égales")

    result = choix_H0_H1(pvalue,confiance)
    return result

def test_distribution_normale(df,confiance = 0.05):
    """
    renvoie une liste des colonne dont la distribution est jugé normale par un test de shapiro
    Ards:
        df : dataframe a tester
        confiance : confiance du test de shapiro
    Returns :
        liste_colonne : liste des colonnes ayant réussies le test
    """
    liste_col_normale = []
    liste_colonne = df.columns.to_list()
    for a in liste_colonne:
        print("\n",a)
        if (test_loi_normale(df[a],confiance)):
            liste_col_normale.append(a)
    return liste_col_normale



######################################################## tests des fontions  #########################################################################

if __name__ == '__main__':
    ############################### partie stat   ##########################""

    #example de courbe de Lorentz (utilisation)

    fig,ax =plt.subplots()
    courbe_Lorentz(ax,df,'A')

    plt.show()

    #exemple de calcul de gini(utilisation)

    #example de test de chi2 (utilisation non conforme a cause du dataset)
    print("########################### test chi2   ##########################")
    test_chi2(df=df2,col1='Nom',col2='type')
