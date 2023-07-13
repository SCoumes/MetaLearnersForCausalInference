import pandas as pd
import numpy as np

def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):
    trainingData.reset_index(drop=True, inplace=True)
    predictionData.reset_index(drop=True, inplace=True)
    treatments = getTreatments(trainingData)
    r = estimate_r(trainingData, treatments, Xcolumns)
    r[np.where(r == 0)] = 1 # Change all 0 to a 1 in r to avoid dividing by 0. This changes nothing, as all calculations where this intervene will yield 0 anyway.
    X_test = predictionData[Xcolumns]
    Mlist = [] 
    for index, _ in enumerate(treatments):
        Mlist.append(trainModel(trainingData, treatments, index, r, Xcolumns=Xcolumns))

    output = predictionData.copy() # Avoid contamination of input target
    N = len(treatments)
    
    for k1 in range(0,N):
        for k2 in range(k1 +1, N):
            output["T" + str(k1) + "T" + str(k2)] = Mlist[k2].predict(X_test) - Mlist[k1].predict(X_test)
    return output
    return pd.DataFrame(columns=["date","daily_sales_with_noise","nb_shops","seasonality","covid_strength","mean_suggested_price","equational_sales", "revenue","weekly_smoothed_sales_with_noise","T0T1","T0T2","T0T3","T0T4","T1T2","T1T3","T1T4","T2T3","T2T4","T3T4"])

def getTreatments(df):
    treatments = []
    for _, row in df.iterrows():
        if row["mean_suggested_price"] not in treatments:
            treatments.append(row["mean_suggested_price"])
    return treatments

def estimate_r(df, treatments, Xcolumns):
    r = np.empty((len(df), len(treatments)))
    nbr_t = dict()
    for _, row in df.iterrows():
        #key_t = (row["nb_shops"], row["seasonality"], row["covid_strength"])
        key_t = tuple(row[Xcolumns]) # Create a tuple of the X values. We are counting how many time each combination appears.
        if key_t not in nbr_t:
            nbr_t.update({key_t : [0] * len(treatments)})
        treatment = treatments.index(row["mean_suggested_price"])
        nbr_t[key_t][treatment] += 1
    for key, value in nbr_t.items():
        sumOnT = sum(value)
        nbr_t[key] = [x / sumOnT for x in value]
    for index, row in df.iterrows():
        # key_t = (row["nb_shops"], row["seasonality"], row["covid_strength"])
        key_t = tuple(row[Xcolumns])
        for indexList, valueList in enumerate(nbr_t[key_t]):
            r[index, indexList] = valueList
    # with np.printoptions(threshold=np.inf):
    #     print(r)
    return r

def trainModel(trainingData, treatments, treatmentIndex, r, model=None, Xcolumns=None):
    from sklearn.ensemble import RandomForestRegressor
    Z = trainingData["revenue"] / r[:,treatmentIndex]
    for index, row in trainingData.iterrows():
        if not row["mean_suggested_price"] == treatments[treatmentIndex]:
            Z[index] = 0
    
    X = trainingData[Xcolumns]
    regr = RandomForestRegressor(max_depth=2, random_state=0) 
    regr.fit(X, Z)
    return regr



# M_Learner <- function(X, Y, W, r_hat, model = c("randomForest", "xgboost", "lm"), p = 3){
#   W.levels <- sort(unique(W))
#   M.fit <- list()
#   for(k in 1:length(W.levels)){
#     w = W.levels[k]
#     Z_w = as.numeric(W == w)*Y/r_hat[,k]
#     if(model == "xgboost"){
#       M.fit[[k]] <- xgboost(data = as.matrix(sapply(X, as.numeric)), 
#                             label = Z_w, nrounds = 100, verbose = FALSE)
#     }else if(model == "randomForest"){
#       M.fit[[k]] <- randomForest(x = as.matrix(X), y = Z_w)
#     }else{
#       M.fit[[k]] <- lm(Z_w ~ polym( as.matrix(X), degree = p, raw = T))
#     }
    
#   }
#   return(M.fit)
# }
