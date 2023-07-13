import pandas as pd

def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):
    """trainingData is a dataframe with exactly the same colmuns as the dataset we received with the addition of a "revenue" column. Likewise for predictionData. 
    Xcolumns is a list containing the names of the columns to use as cofounders.
    This function should return a dataframe with the same columns as the example shown bellow and a line per line in predictionData. 
    For example, the value of T0T1 is the value obtained with treatment T1 - the value obtained with treatment T0."""
    return pd.DataFrame(columns=["date","daily_sales_with_noise","nb_shops","seasonality","covid_strength","mean_suggested_price","equational_sales", "revenue","weekly_smoothed_sales_with_noise","T0T1","T0T2","T0T3","T0T4","T1T2","T1T3","T1T4","T2T3","T2T4","T3T4"])
