import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from econml.metalearners import SLearner
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from MetaLearners.TrueLearnerExample import trainAndPredict as tpTruelearner

sns.set_theme()
# For the image quality of the graphic. 
sns.set(rc={"figure.dpi":300})
# For the size of the graphics
sns.set(rc = {"figure.figsize":(6,3)})


from sklearn.model_selection import train_test_split



def trainAndPredict(trainingData, predictionData, Xcolumns, model=None):
    # Generate a true_learner_df using the tpTruelearner function
    true_learner_df = tpTruelearner(trainingData, predictionData, Xcolumns, model)
     # Extract features and target variables from the trainingData
    X = trainingData[Xcolumns]  
    X_test = predictionData[Xcolumns]  
    T = trainingData.mean_suggested_price  
    Y = trainingData.revenue 
     # Get unique treatments from the mean_suggested_price column and sort them
    treatments = sorted(trainingData.mean_suggested_price.unique()) 
     # Calculate the number of samples and number of unique treatments
    n = len(X)  
    N = len(treatments)
     # Create a list of column names for training data
    train_cols = Xcolumns + ["mean_suggested_price"]
     # Generate synthetic data with the generate_data function
    df = generate_data(trainingData, train_cols, 1000000)
     # Remove duplicate rows from the generated data
    df = df.drop_duplicates()
     # Train a machine learning model on the generated data
    model_fit = train_ml(df, Xcolumns)
     # Define a function to calculate the effect of different treatments on predictions
    def effect_txty(model_fit, data_df, T0, T1):
        return model_fit.predict(data_df.assign(mean_suggested_price=T0)) * 10e3 - model_fit.predict(data_df.assign(mean_suggested_price=T1)) * 10e3
     # Create a DataFrame containing the effect of each treatment pair on predictions
    S_tes = pd.DataFrame([effect_txty(model_fit, X_test, treatments[i], treatments[j])
                          for i in range(len(treatments)) 
                          for j in range(i+1, len(treatments))], index=[f"T{i}T{j}" for i in range(N) for j in range(i+1, N)]).T
     # Concatenate the predictionData and S_tes DataFrames
    output_df = pd.concat([predictionData.reset_index(drop=True, inplace=False), S_tes], axis=1)
     # Return the output DataFrame
    return output_df



def train_ml(data_df,x_cols):
    model = RandomForestRegressor(**{'n_estimators': 200,
    'bootstrap': False})
    # model = DecisionTreeRegressor()


    train_cols= x_cols+["mean_suggested_price"]
    X_train, X_test, y_train, y_test = train_test_split(data_df[train_cols], data_df.revenue, test_size=0.05, random_state=42)

    model_fit = model.fit(X_train,y_train)

    model_fit.score(X_test,y_test)

    prediction = model_fit.predict(data_df[train_cols])
    mse = mean_squared_error(data_df.revenue, prediction)
    rmse = mse**.5
    print(mse)
    print(rmse)

    return model_fit


def plot_predictions(model_fit,data_df,x_cols,train_cols,treatments,label="withoutgeneration"):
    plt.figure()
    if label =="withgeneration" :

        plt.plot(model_fit.predict(data_df[train_cols])*10e3, label= "prediction" )
    else : 
        plt.plot(model_fit.predict(data_df[train_cols]), label= "prediction" )

    plt.plot(data_df.revenue,"*", label = "revenue")
    plt.xlabel("Number of days")

    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(".\Presentation\images\prediction_"+label+".png")

def plot_CATE(model_fit,data_df,x_cols,train_cols,treatments,true_learner_df,label="withoutgeneration"):
    plt.figure()
    effT0T1 = model_fit.predict(data_df[x_cols].assign(mean_suggested_price = treatments[1])) - model_fit.predict(data_df[x_cols].assign(mean_suggested_price = treatments[0]))

    plt.plot(true_learner_df.T0T1,label ="truelearner")
    if label =="withgeneration" :

        plt.plot(effT0T1*10e3,label = "Classical ML as Slearner")
    else : 
        plt.plot(effT0T1,label = "Classical ML as Slearner")

    plt.xlabel("Number of days")

    plt.ylabel("CATE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(".\Presentation\images\CATE_"+label+".png")


def generate_data(data_df, train_cols, n=1000000):
    # Create an empty DataFrame with n rows
    df = pd.DataFrame([
        np.random.choice(data_df.nb_shops.unique(), n),
        np.random.choice(data_df.seasonality.unique(), n),
        np.random.choice(data_df.covid_strength.unique(), n),
        np.random.choice(data_df.mean_suggested_price.unique(), n)
    ], index=train_cols).T
    
    # Calculate the revenue column using the values from the selected columns
    df = df.assign(revenue=836.8200836820083 * ((df.nb_shops * df.mean_suggested_price * df.seasonality) / df.covid_strength).values / 10e3)
    
    return df


def generate_all(data_df, x_cols, true_learner_df):
    # Create a list of column names for training data
    train_cols = x_cols + ["mean_suggested_price"]
     # Create a list of unique treatments from the mean_suggested_price column
    treatments = sorted(data_df.mean_suggested_price.unique())
     # Train a machine learning model on the original data
    model_fit = train_ml(data_df, x_cols)
     # Plot predictions without generated data
    plot_predictions(model_fit, data_df, x_cols, train_cols, treatments, label="withoutgeneration")
     # Plot CATE without generated data
    plot_CATE(model_fit, data_df, x_cols, train_cols, treatments, true_learner_df, label="withoutgeneration")
     # Generate synthetic data
    df = generate_data(data_df, train_cols, 1000000)
     # Remove duplicate rows from the generated data
    df = df.drop_duplicates()
     # Train a machine learning model on the generated data
    model_fit = train_ml(df, x_cols)
     # Plot predictions with generated data
    plot_predictions(model_fit, data_df, x_cols, train_cols, treatments, label="withgeneration")
     # Plot CATE with generated data
    plot_CATE(model_fit, data_df, x_cols, train_cols, treatments, true_learner_df, label="withgeneration")

    