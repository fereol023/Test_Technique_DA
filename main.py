import os
import time
import warnings
warnings.filterwarnings('ignore')

from joblib import load, dump
import numpy as np
import pandas as pd
from pprint import pprint
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro, pearsonr

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from dstoolbox.pipeline import DataFrameFeatureUnion
from sklearn.pipeline import Pipeline

def line(fonction):
    def wrapper(*args, **kwargs):
        line = '='*200
        print(line)
        start = time.time()
        resultat = fonction(*args, **kwargs)
        print(f"Durée : {round(time.time()-start, 5)} secs")
        return resultat
    return wrapper

class Explore:

    def __init__(self, csvfilepath, target='TotalCart'):
        self.target = target
        self.path = csvfilepath
        self.df = pd.read_csv(csvfilepath)
        
    @line
    def globalDescription(self):
        print(self.df.dtypes)
        print(self.df.describe())

    @line
    def globalCheckNull(self):
        for col in self.df.columns:
            print(f"Null counts {col} : {self.df[col].isnull().sum()}")

    @line
    def distroTopCategory(self):
        # nbre de modalités
        for col in self.df.select_dtypes(include='object').columns:
            print(f"{col} has {self.df[col].nunique()} values")
            print(100*self.df[col].value_counts()/self.df[col].count())
        # graphes

    @line
    def normality(self):
        # tests de normalité 
        shapiro_res = {}
        for col in self.df.select_dtypes(exclude='object').columns:
            shapiro_res.update({col: shapiro(self.df[col])[1]})
            self.shapiro_df = pd.DataFrame.from_dict(shapiro_res, orient='index', columns=['pvalue'])
        print(self.shapiro_df)
        # graphes

    @line
    def correlation(self):
        # tests de correlation 
        corr_res = []
        for col1,col2 in list(product(self.df.columns, self.df.columns)):
            try:
                corr, xx = pearsonr(self.df[col1], self.df[col2])
                corr_res.append((col1, col2, corr, xx))
            except Exception as e:
                pass
        corr_res = [x for x in corr_res if x[0] == self.target and x[3] <= .05]
        pprint(corr_res)

# Feature engineering
class QuantitativeEncoder(BaseEstimator, TransformerMixin):
    """
    Renvoie un pd df restreint sur les variables quanti.
    Ne fait rien.
    """
    def __init__(self, columns = None):
        super().__init__()
        self.columns = columns

    def fit(self, X, y = None):
        if self.columns is not None:
            self.X = X[self.columns]
        return self
    
    def transform(self, X):
        if self.columns is not None:
            X = X[self.columns]
        return X

class QualitativeEncoder(BaseEstimator, TransformerMixin):
    """
    Renvoie un pd dataframe au lieu d'un <class 'numpy.ndarray'>. 
    Sauvegarde l'encodeur pour la reproductibilité. 
    """
    def __init__(self, columns = None):
        super().__init__()
        self.columns = columns
        if os.path.exists('./model/artifacts/fitted_labelEncoder.joblib'):
            self.labencode = load('./model/artifacts/fitted_labelEncoder.joblib')
        else:
            self.labencode = LabelEncoder()
        
    def fit(self, X, y = None):
        if self.columns is not None:
            self.X = X[self.columns]

        self.labencode.fit(self.X)
        dump(self.labencode, './model/artifacts/fitted_labelEncoder.joblib')
        return self

    def transform(self, X):
        if self.columns is not None:
            X = X[self.columns]
        return pd.DataFrame(self.labencode.transform(X), columns = self.columns)

class ML_pipeline:
    def __init__(self, model=None):
        self.model = model
        quant = [
            'Age',
            'Seniority',
            'Orders',
            'Items',
            'AverageDiscount',
            'BrowsingTime',
            'EmailsOpened',
            'SupportInteractions'
        ]
        self.feature_pipeline = DataFrameFeatureUnion(
            [
                ('QuantitativesEncoding', QuantitativeEncoder(columns = quant)),
                ('TopCategoryEncoding', QualitativeEncoder(columns = ['TopCategory']))
            ]
        )

    def build_pipeline(self):
        """
        Pour construire le pipeline.
        """
        ML_pipeline_instance = Pipeline(
            [
                ('FeatureEngineering', self.feature_pipeline),
                ('Model', self.model)
            ]
        )
        return ML_pipeline_instance  


# utiliser la classe ML pipeline pour entrainer avec la CV le meilleur modèle + sauvegarder
# charger la sauvegarde et l'évaluer sur de nouvelles p_data
@line
def main(p_data, mode='save_cv'):
    """
    Evaluer le modele sur une p data.
    mode = save_cv/evaluation
    """

    x_features_names = [
        'Age',
        'Seniority',
        'Orders',
        'Items',
        'AverageDiscount',
        'BrowsingTime',
        'EmailsOpened',
        'SupportInteractions',
        'TopCategory'
    ]
    x_features = p_data[x_features_names]
    y_target = p_data['TotalCart']
    
    if mode == 'save_cv':

        param_rf = {
            'n_estimators' : [int(x) for x in np.linspace(10, 80, 10)], # nombre d'arbres dans la forêt 10 20 30 ..
            'max_depth' : [2, 1000], # nombre max de niveaux dans un arbre 
            'min_samples_split' : [5, 10], # nombre min d'echantillons (bootstrap) necessaire au niveau d'un noeud pour juger de le spliter 
            'min_samples_leaf' : [1, 200], # nombre min de samples requis à chaque node 
            'max_features' : ['auto', 'sqrt'] # nombre de features à considérer auto = et sqrt = 
        }

        rfcv = GridSearchCV(RandomForestRegressor(), 
                                param_grid=param_rf, 
                                cv = 4, 
                                n_jobs = 10)

        ML_pipeline(model=rfcv).build_pipeline().fit(x_features, y_target)
        dump(rfcv.best_estimator_, './model/outputs/rf_best_model.joblib') 
        print(f'Model saved with params : {rfcv.best_params_}')

    elif mode == 'evaluation':
        best_rfcv = load('./model/outputs/rf_best_model.joblib')
        y_preds = ML_pipeline(model=best_rfcv).build_pipeline().predict(x_features)
        rmse = round(mean_squared_error(y_target, y_preds, squared=False), 3)
        print(f'Evaluation mean error : {rmse}')

    else:
        print("Mode invalide.")

if __name__=='__main__':

    p0_path = "./datasets/period_0.csv"
    p1_path = "./datasets/period_1.csv"
    p2_path = "./datasets/period_2.csv"
    p3_path = "./datasets/period_3.csv"
    #Explore(p0_path).globalDescription()
    #Explore(p0_path).globalCheckNull()
    Explore(p0_path).distroTopCategory()
    Explore(p1_path).distroTopCategory()
    Explore(p2_path).distroTopCategory()
    Explore(p3_path).distroTopCategory()

    #Explore(p0_path).normality()
    #Explore(p0_path).correlation()

    p0 = Explore(p0_path).df
    p1 = Explore(p1_path).df
    p2 = Explore(p2_path).df
    p3 = Explore(p3_path).df

    p0_train_validate = p0.head(800)
    p0_evaluation = p0.tail(200)
    main(p_data=p0_train_validate, mode='save_cv')
    main(p_data=p0_evaluation, mode='evaluation')
    main(p_data=p1, mode='evaluation')
    main(p_data=p2, mode='evaluation')
    main(p_data=p3, mode='evaluation')

    # conclusion
    # Le CA est fortement corrélé avec le tps passé sur le site. 
    # Ttes les variables quantitatives sont issues de distributions normales y.c. la target.
    # 1/4 des TopCateg sont des Beauty & Personal Care et 1/4 des Clothings shoes et jewelry. 


