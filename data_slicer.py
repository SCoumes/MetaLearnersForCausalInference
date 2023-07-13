# TODO: ignore "nb_shops"
# TODO: slice by "covid_strength"
# TODO: keep "seasonality"

import pandas as pd
import numpy as np

from os.path import join, dirname, exists
from os import getcwd, makedirs

from runAllLearners import run_all_learners

# Constants
IGNORE_NB_SHOPS = False  # ignore nb_shops
SLICING_COL = "years"  # slice by covid_strength
REVENUE = "alternate"  # use equational sales as revenue for now
NOISY = True  # add noise to revenue


def alternate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with revenue values: revenue = daily_sales * mean_suggested_price
    
    ---
    This is an alternative revenue definition different from the one used in the MMM challenge
    """
    df = df.copy()
    df["revenue"] = (
        df["daily_sales_with_noise"] * df["mean_suggested_price"]
    )  # alternate revenue
    return df


def shame_revenue(df: pd.DataFrame, coeff: float = 1.0) -> pd.DataFrame:
    """
    With this definition of revenue, we should get a CATE of 0 for each learner
    """
    df = df.copy()
    if coeff == 1.0:
        coeff = (
            df["equational_sales"]
            / (df["nb_shops"] * df["seasonality"])
            * df["mean_suggested_price"]
            / df["covid_strength"]
        ).mean()
        print("coeff = ", coeff)
    df["revenue"] = coeff * (df["nb_shops"] * df["seasonality"]) / df["covid_strength"]
    return df


def noisy_revenue(df: pd.DataFrame, var: float = 0.15) -> pd.DataFrame:
    """
    Returns a dataframe with revenue values that are the same as the mean revenue
    """
    df = df.copy()
    df["revenue"] = df["revenue"] * (1 + np.random.normal(0, var, len(df)))
    return df


# x_cols = ["seasonality", "nb_shops", "covid_strength"]

# # x_cols = ["seasonality"]  # ignore "nb_shops" and "covid_strength"


def getFinalOutputPath(outputPath: str, **kwargs) -> str:
    """The function `getFinalOutputPath` takes an `outputPath` and optional keyword arguments, and returns
    a modified `outputPath` based on the values of the keyword arguments.

    Parameters
    ----------
    outputPath : str
        The initial output path as a string.

    Returns
    -------
        the final output path  where the final output will be saved as a string.

    """
    try:
        # get the arguments from kwargs ;
        # this design choice was made to make the function more flexible ;
        # the function can be called with any combination of arguments as long as the names match
        SLICING_COL = kwargs.get("slicing_col")
        REVENUE = kwargs.get("revenue")
        NOISY = kwargs.get("noisy")
    except KeyError as e:
        print("KeyError: ", e)
        return outputPath

    if SLICING_COL == "covid_strength":
        outputPath = join(outputPath, "covid_sliced")
    elif SLICING_COL == "years":
        outputPath = join(outputPath, "year_sliced")
    else:
        raise ValueError("SLICING_COL must be 'covid_strength' or 'years'")

    if REVENUE == "alternate":
        outputPath = join(outputPath, "alternate_revenue")
    elif REVENUE == "shame":
        outputPath = join(outputPath, "shame_revenue")

    if NOISY:  # add noise to our revenue as created by the above revenue definitions
        outputPath = join(outputPath, "noisy_revenue")
    else:
        outputPath = join(outputPath, "clean_revenue")

    if not exists(outputPath):  # make the output path if it doesn't exist
        makedirs(outputPath)

    return outputPath


def run_slicer(**kwargs) -> None:
    """The `run_slicer` function takes in various keyword arguments, including a dataframe (`df`), and
    performs slicing and modeling operations based on the provided arguments.

    Returns
    -------
        The function `run_slicer` does not return any value. It performs various operations based on the
    input arguments provided and prints the results.

    """

    try:
        # get the arguments from kwargs ;
        # this design choice was made to make the function more flexible ;
        # the function can be called with any combination of arguments as long as the names match
        df = kwargs.get("df")
        x_cols = (
            kwargs.get("x_cols")
            if "x_cols" in kwargs
            else ["seasonality", "nb_shops", "covid_strength"]
        )  # default x_cols
        revenue = (
            kwargs.get("revenue") if "revenue" in kwargs else "equational_sales"
        )  # default revenue
        noisy = kwargs.get("noisy") if "noisy" in kwargs else False  # default noisy
        ignore_nb_shops = (
            kwargs.get("ignore_nb_shops") if "ignore_nb_shops" in kwargs else False
        )  # default ignore_nb_shops
        SLICING_COL = (
            kwargs.get("slicing_col") if "slicing_col" in kwargs else "years"
        )  # default slicing_col
        outputPath = kwargs.get("outputPath")
    except KeyError as e:
        print("KeyError: ", e)
        return None

    outputPath = getFinalOutputPath(
        outputPath=outputPath, revenue=revenue, noisy=noisy, slicing_col=SLICING_COL
    )

    if (
        ignore_nb_shops
    ):  # along the experiment, we wanted to see what happens if we ignore "nb_shops"
        x_cols.remove("nb_shops")
    if SLICING_COL in x_cols:
        x_cols.remove(SLICING_COL)

    if revenue == "alternate":
        df = alternate_revenue(df)
    elif (
        revenue == "shame"
    ):  # the name "shame" is to remind us that this false revenue definition should give a CATE of 0 for each learner
        df = shame_revenue(df)

    if noisy:
        df = noisy_revenue(df)
    # DRY: Don't Repeat Yourself
    # The following code does not respect the DRY principle, but it is the easiest way to run the experiments
    if SLICING_COL == "covid_strength":
        covid_strengths = np.unique(df["covid_strength"])
        print("slices:  ", covid_strengths)
        for val in covid_strengths:
            df_slice = df[df["covid_strength"] == val]
            if len(df_slice.mean_suggested_price.unique()) == 1:
                print(f"Skipping {val} because there is only one mean_suggested_price")
                continue
            print(f"Running {val}")
            run_all_learners(
                df_slice,
                None,  # run all models
                x_cols,
                outputPath,
                outputFile=f"_{val:03f}_Predictions",
                verbose=False,
            )
            print(f"Done with {val}")

    df["years"] = df["date"].apply(lambda x: int(x.split("-")[0]))

    if SLICING_COL == "years":
        years = np.unique(df["years"])
        print("slices:  ", years)
        for val in years:
            df_slice = df[df["years"] == val]
            if len(df_slice.mean_suggested_price.unique()) == 1:  # skip if constant
                print(f"Skipping {val} because there is only one mean_suggested_price")
                continue
            print(f"Running {val}")
            run_all_learners(
                df_slice,
                None,  # run all models
                x_cols,
                outputPath,
                outputFile=f"_{val:d}_Predictions",
                verbose=False,
            )
            print(f"Done with {val}")


if __name__ == "__main__":
    dataPath = join(
        join(join(dirname(getcwd()), "data"), "eki"), "mmm_challenge_like.csv"
    )
    df = pd.read_csv(dataPath)
    df["revenue"] = df["equational_sales"]  # use equational sales as revenue for now
    outputPath = join(join(dirname(getcwd()), "data"), "output")
    if not exists(outputPath):
        makedirs(outputPath)

    print("Saving files to: ", outputPath)
    for noisy in [True, False]:
        print("-" * 25)
        print("Noisy: ", noisy)
        print("-" * 25)
        for revenue in ["alternate", "shame", "equational_sales"]:
            print("-" * 20)
            print("Revenue: ", revenue)
            print("-" * 20)
            run_slicer(
                df=df,
                revenue=revenue,
                ignore_nb_shops=IGNORE_NB_SHOPS,
                slicing_col=SLICING_COL,
                outputPath=outputPath,
                noisy=noisy,
            )
