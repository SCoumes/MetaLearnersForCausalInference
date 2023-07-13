from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os.path import join, dirname, exists

from os import getcwd, makedirs
import warnings

import seaborn as sns
sns.set_theme()
# For the image quality of the graphic. 
sns.set(rc={"figure.dpi":300})
# For the size of the graphics
sns.set(rc = {"figure.figsize":(6,3)})


# Repeat the following for all models

# example learner
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from MetaLearners.learnerExample import trainAndPredict as tpExample
    from MetaLearners.SLearnerExample import trainAndPredict as tpSlearner
    from MetaLearners.TLearnerExample import trainAndPredict as tpTlearner
    from MetaLearners.XLearnerExample import trainAndPredict as tpXlearner
    from MetaLearners.DALearnerExample import trainAndPredict as tpDAlearner
    from MetaLearners.DRLearnerExample import trainAndPredict as tpDRlearner
    from MetaLearners.RLearnerExample import trainAndPredict as tpRlearner
    from MetaLearners.MLearner import trainAndPredict as tpMlearner
    from MetaLearners.TrueLearnerExample import trainAndPredict as tpTruelearner
    from MetaLearners.MLclassicalLearner import generate_all, trainAndPredict as MLclearner
MODELS = {
    # tpExample: "tpExample", # example learner
    tpSlearner: "tpSlearner",
    tpTlearner: "tpTlearner",
    tpXlearner: "tpXlearner",
    tpDAlearner: "tpDAlearner",
    tpDRlearner: "tpDRlearner",
    tpRlearner: "tpRlearner",
    tpMlearner: "tpMlearner",
    tpTruelearner: "tpTruelearner",
}


def run_all_learners(
    df: pd.DataFrame | np.ndarray,
    model_list: dict|None,
    xcols: list[str],
    outputPath: str,
    outputFile: str = "Predictions",
    verbose: bool = True,
) -> None:
    """
    Runs all learners in model_list and saves the predictions to outputPath
    """
    global MODELS
    if model_list is None:  # if model_list is None, run all models
        model_list = MODELS
    if not exists(outputPath):
        makedirs(outputPath)

    for m in model_list:
        model_name = model_list.get(m)
        print(f"Running {model_name}") if verbose else None
        
        ### The following is a hack to deal with ZeroDivisionError that randomly occurs with some learners (DA_learner in particular)
        max_attempts = 3  # Maximum number of attempts
        attempts = 0  # Counter variable
        while attempts < max_attempts:
            try:
                outputExample = m(df, df, xcols)
                break
            except ZeroDivisionError as e:
                print(f"Attempt {attempts + 1} failed for {model_name} with exception: {e}")
                attempts += 1
        else:
            print("Maximum number of attempts reached.")
            continue
        ### End of hack
        
        #print(outputExample) if verbose else None
        outputExample.to_csv(
            join(outputPath, f"{model_name}{outputFile}.csv"), sep="\t"
        )
        print(f"Done with {model_name} \n") if verbose else None

if __name__ == "__main__":
    dataPath = join(getcwd(), "data.csv")
    df = pd.read_csv(dataPath)
    df["revenue"] = df["equational_sales"]

    outputPath = join(getcwd(), "output")
    if not exists(outputPath):
        makedirs(outputPath)

    print("Saving files to: ", outputPath)

    # colmuns_to_plot = ['nb_shops', 'seasonality', 'covid_strength', 'mean_suggested_price', 'revenue']
    # for col in colmuns_to_plot:
    #     plt.figure()
    #     df[col].plot()
    #     # plt.xticks(df.index[::60],df.date.astype("str")[::60],rotation=90);
    #     plt.xlabel("number of days")
    #     plt.ylabel(col)
    #     plt.tight_layout()
    #     # path_for_plot = join(join(join(dirname(getcwd()), "Presentation"), "images"),"viz"+col+".png")
    #     plt.savefig(".\Presentation\images\Viz_"+col+".png")
    #     print("hello")
    # similar to calling run_all_learners(df, None, x_cols, outputPath)
    
    x_cols = ["nb_shops", "seasonality", "covid_strength"]
    run_all_learners(df, None, x_cols, outputPath)
    
    ##### The two following lines run the truelearner directly to plot results. Without them, the true learner is still called by run_all_learners anyway. The two following lines are commented because they otherwise produce plots that are no longer obviously required.
    # true_learner_df = tpTruelearner(df, df, x_cols)
    # generate_all(df,x_cols,true_learner_df)