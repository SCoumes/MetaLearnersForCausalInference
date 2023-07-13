import pandas as pd
from econml.dr import DRLearner
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):

    X  = trainingData[Xcolumns]

    X_test = predictionData[Xcolumns]
    T = trainingData.mean_suggested_price
    Y = trainingData.revenue
    treatments = sorted(trainingData.mean_suggested_price.unique())
    n=len(X)
    N=len(treatments)
    outcome_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
    pseudo_treatment_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
    propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                                      min_samples_leaf=max(1, int(n/100)))

    DR_learner = DRLearner(model_regression=outcome_model, model_propensity=propensity_model,
                          model_final=pseudo_treatment_model, cv=5)
    DR_learner.fit(Y, T, X=X)


    DR_tes= pd.DataFrame([DR_learner.effect(X_test,T0=treatments[i],T1=treatments[j])
                          for i in range(len(treatments)) 
                          for j in range(i+1,len(treatments))],index=[f"T{i}T{j}" for i in range(N) for j in range(i+1, N)]).T
    output_df = pd.concat([predictionData.reset_index(drop=True, inplace=False),DR_tes],axis=1)
    return output_df
