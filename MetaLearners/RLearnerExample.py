import numpy as np
import pandas as pd
from causalml.inference.meta import BaseRRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


def trainAndPredict(trainingData, predictionData, xcols, model=None):

    X : pd.DataFrame | np.ndarray = trainingData[xcols]

    X_test : pd.DataFrame | np.ndarray = predictionData[xcols]
    T : pd.Series | np.ndarray = trainingData.mean_suggested_price
    Y : pd.Series | np.ndarray = trainingData.revenue
    treatments : list = sorted(trainingData.mean_suggested_price.unique())
    n : int = len(X)

    if model is None :
        model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
    propensity_model=RandomForestClassifier(n_estimators=100, max_depth=6,
                                                      min_samples_leaf=max(1, int(n/100)))
    R_learner = BaseRRegressor(model, control_name='control', propensity_learner=propensity_model)
    
    R_tes = {}
    
    for idx, i in enumerate(treatments):
        treatment = np.array([val if val!=i else 'control' for val in T.to_numpy()])
        
        R_learner.fit(X, treatment, Y)
        
        R_tes[i] = R_learner.predict(X_test)[:, idx:]
        
    R_tess = pd.concat([pd.DataFrame(R_tes[i]) for i in R_tes.keys()], axis=1)
    N = len(treatments)
    R_tess.columns = [f"T{i}T{j}" for i in range(N) for j in range(i+1, N)]
    output_df = pd.concat([predictionData.reset_index(drop=True, inplace=False),R_tess],axis=1)
    return output_df
