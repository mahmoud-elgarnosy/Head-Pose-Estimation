from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class noramlize_points(BaseEstimator, TransformerMixin):
    def fit(self, X,y = None):
        return self
    def transform(self, X,y = None):

        #Divide all the points by the Subtraction of borow and nose
        brow_nose_dist = np.sqrt(((X.iloc[:,296].values - X.iloc[:,1].values) **2) + ((X.iloc[:,792].values - X.iloc[:,470].values) **2))
        X = X / brow_nose_dist[:,np.newaxis]
        
        #Subtract all points form nose points(x,y)
        X.iloc[:,:468] = X.iloc[:,:468].values- X[1].values[:,np.newaxis]
        X.iloc[:,469:] = X.iloc[:,469:].values- X[470].values[:,np.newaxis]
        
        return X