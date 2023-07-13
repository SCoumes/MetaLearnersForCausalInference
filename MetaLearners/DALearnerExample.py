import pandas as pd
from econml.metalearners import DomainAdaptationLearner
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):

    X  = trainingData[Xcolumns]

    X_test = predictionData[Xcolumns]
    T = trainingData.mean_suggested_price
    Y = trainingData.revenue
    treatments = sorted(trainingData.mean_suggested_price.unique())
    n=len(X)
    N=len(treatments)   
    models = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
    final_models = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
    propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                                  min_samples_leaf=max(1, int(n/100)))
    DA_learner = DomainAdaptationLearner(models=models,
                                     final_models=final_models,
                                     propensity_model=propensity_model)
    DA_learner.fit(Y, T, X=X)


    D_tes= pd.DataFrame([DA_learner.effect(X_test,T0=treatments[i],T1=treatments[j])
                          for i in range(len(treatments)) 
                          for j in range(i+1,len(treatments))],index=[f"T{i}T{j}" for i in range(N) for j in range(i+1, N)]).T
    output_df = pd.concat([predictionData.reset_index(drop=True, inplace=False),D_tes],axis=1)
    return output_df
