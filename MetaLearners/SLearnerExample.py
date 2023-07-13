import pandas as pd
from econml.metalearners import SLearner
from sklearn.ensemble import GradientBoostingRegressor
def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):
    """
    Trains an S-learner model using the training data and applies the trained model to predict treatment effects for the prediction data.
    Parameters:
        trainingData (DataFrame): The data used for training the model.
        predictionData (DataFrame): The data used for prediction.
        Xcolumns (list): List of column names used as features for training and prediction.
        model (object, optional): The model object to be used. If None, a default GradientBoostingRegressor is used.
    Returns:
        output_df (DataFrame): The prediction results of treatment effects.
    """
    X = trainingData[Xcolumns]  
    X_test = predictionData[Xcolumns]  
    T = trainingData.mean_suggested_price  
    Y = trainingData.revenue 
    treatments = sorted(trainingData.mean_suggested_price.unique()) 
    n = len(X)  
    N = len(treatments) 
    if model is None:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)))
        # Use default GradientBoostingRegressor if no model is provided
    S_learner = SLearner(overall_model=model)  # Initialize the S-learner with the chosen model
    S_learner.fit(Y, T, X=X)  # Fit the S-learner model using the training data
    S_tes = pd.DataFrame([S_learner.effect(X_test, T0=treatments[i], T1=treatments[j])
                        for i in range(len(treatments))
                        for j in range(i + 1, len(treatments))],
                        index=[f"T{i}T{j}" for i in range(N) for j in range(i + 1, N)]).T
    # Calculate the treatment effects for all pairs of treatments
    output_df = pd.concat([predictionData.reset_index(drop=True, inplace=False), S_tes], axis=1)
    # Combine predictionData and treatment effects
    return output_df