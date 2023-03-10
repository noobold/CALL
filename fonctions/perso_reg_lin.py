import numpy as np
import pandas as pd

from sklearn import linear_model

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score ,max_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import fonctions.perso_stats as perso_stats

def regression_lineaire(data_train,data_test,target_train,target_test,intercept=True):
    """
    crée un modele de regression lineaire.
    arg :
        train : dataframe ou serie ou array données d entrainement         
        target dataframe ou serie ou array         
        
    return :
        regr : model sklearn de regression lineaire
    """
    #sklearn version
    #creation model regression lineaire
    regr = linear_model.LinearRegression(fit_intercept=intercept) #version sklean
    
    #entrainement du model
    regr.fit(data_train,target_train)    
    
    #prediction du modele pour test
    pred_train = regr.predict(data_train)
    
    if intercept:
        print("intercept (const dans statmodels)",regr.intercept_)

    # The coefficients
    print("Coefficients: \n", regr.coef_)    
    # The mean squared error
    print("Erreur des moindres carrés train : %.2f" % mean_squared_error(target_train, pred_train))
    # The coefficient of determination: 1 is perfect prediction
    # https://stackoverflow.com/questions/54614157/scikit-learn-statsmodels-which-r-squared-is-correct
    print("Coefficient de determination train : %.2f" % r2_score(target_train, pred_train))
    #erreur max :
    print("erreur max train: ",max_error(target_train, pred_train))
    #score target et score test pour surapprentissage
    print("score train :",regr.score(data_train,target_train))
    print("score test :",regr.score(data_test,target_test))

    
    fig,ax = plt.subplots()
    erreur_train =pred_train-target_train
    sns.histplot(erreur_train,ax=ax)
    ax.set_title("distribution des erreurs")    
    
    plt.show()

    #test normalité
    print("\n test de Normalité des erreurs")
    perso_stats.test_loi_normale(erreur_train)
    print("\n")
    #statmodel integre par intercept par defaut faut l ajouter
    train_ = data_train
    if intercept:
        train_ = sm.add_constant(data_train)
    else :
        train_ = data_train
    
    model = sm.OLS(target_train,train_)
    result = model.fit()
    print (result.summary())

    

    return(regr)

def auto_regression_lineaire(data,colonne_cible,intercept=True):
    """
    crée un modele de regression lineaire
    arg:
        data : dataframe de donnée
        colonne_cible: string nom de la colonne cible
    return:
        regr : modele sklearn de regression lineaire
    """
    data_train,data_test = train_test_split(data)
    target_train = data_train.pop(colonne_cible)
    target_test = data_test.pop(colonne_cible)

    #reshape si series (sklean n aime pas trop les pandas series)
    
    if isinstance(data_train,pd.Series):
        data = np.array(data)
        data =data.reshape(-1,1)

    if isinstance(data_test,pd.Series):
        data = np.array(data)
        data =data.reshape(-1,1)
 
        
    return regression_lineaire(data_train,data_test,target_train,target_test,intercept=intercept)

