import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join, dirname, exists
from os import getcwd, makedirs


def plot_all_treatments(output_folder):
    """
    Plots the CATE values for all treatments from CSV files in the given output folder.

    Args:
        output_folder (str): Path to the output folder containing the CSV files.

    """
    files = os.listdir(output_folder)
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(output_folder, file), delimiter="\t")
            filename = os.path.splitext(file)[0]
            fig, ax = plt.subplots(figsize=(15, 9), dpi=80)
            columns_to_plot = [
                "T0T1",
                "T0T2",
                "T0T3",
                "T0T4",
                "T1T2",
                "T1T3",
                "T1T4",
                "T2T3",
                "T2T4",
                "T3T4",
            ]
            data_to_plot = df[columns_to_plot]
            ax.plot(data_to_plot)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("CATE")
            plt.title(f"{filename[2:-11]} CATE values")
            ax.legend(columns_to_plot, loc="lower left")
            fig.savefig(os.path.join(output_folder, f"{filename}_figure.png"))
            plt.show()


def plot_all_learners_separate(output_folder, column_name="T0T1"):
    """
    Plots the CATE values for each learner from separate CSV files in the given output folder.

    Args:
        output_folder (str): Path to the output folder containing the CSV files.
        column_name (str): The name of the column to plot. Default is 'T0T1'.
    """
    files = [file for file in os.listdir(output_folder) if file.endswith(".csv")]
    fig = plt.figure(figsize=(15, 9), dpi=80)
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(output_folder, file), delimiter="\t")
        filename = os.path.splitext(file)[0]
        columns_to_plot = [column_name]
        data_to_plot = df[columns_to_plot]
        ax = fig.add_subplot(len(files), 1, i + 1)

        ax.plot(data_to_plot, label=filename[:-11])
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("CATE")
        # ax.set_title(f'{filename[2:-11]} CATE values')
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "combined_files_figure.png"))
    plt.show()


def plot_all_learners_combined(
    output_folder: str, column_name="T0T1", sliced: bool = False
):
    """
    Plots the CATE values for all learners except for Rlearner on 1 Fig.

    Args:
        output_folder (str): Path to the output folder containing the CSV files.
        columns_to_plot (str or list): The name(s) of the column(s) to plot. Default is 'T0T1'.
    """
    files = [file for file in os.listdir(output_folder) if file.endswith(".csv")]
    fig, ax = plt.subplots(figsize=(15, 9), dpi=80)

    for index, file in enumerate(files):
        # if index in [0, 1, 2, 3, 5]:  # skip Rlearner
        if file.startswith("tpRlearner"):
            continue
        df = pd.read_csv(os.path.join(output_folder, file), delimiter="\t")
        filename = os.path.splitext(file)[0]
        try:
            data_to_plot = df[column_name]
        except KeyError:
            print(f"Column {column_name} not found in {filename}")
            print(f"Available columns: {df.columns}")
            continue
        ax.plot(data_to_plot, label=filename[2:-11])

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("CATE")
    plt.title("CATE values for Different Learners")
    ax.legend(loc="lower left")
    fig.savefig(os.path.join(output_folder, "combined_files_figure.png"))
    plt.show()


def describe_data(data_file, output_dir=None):
    """Produces plots for data exploratory analysis"""
    plt.rcParams["figure.figsize"] = (10, 6)
    df = pd.read_csv(data_file)

    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Plot Price vs. Covid Strength
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["covid_strength"], color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Covid Strength", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(df["mean_suggested_price"], color="red")
    plt.title("Mean Suggested Price vs. Covid Strength", fontsize=14, fontweight="bold")
    save_file = "price_vs_covid.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)  # Save the figure as a PNG image with 300 DPI
    plt.show()

    # Plot Daily Sales with Noise
    plt.plot(df["date"], df["daily_sales_with_noise"], color="blue", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Daily Sales with Noise")
    plt.title("Daily Sales with Noise", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    # Optional: Set the x-axis limits based on the date range
    plt.xlim(df["date"].min(), df["date"].max())
    save_file = "daily_sales_with_noise.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)
    plt.show()

    # Plot Number of Shops
    plt.plot(df["date"], df["nb_shops"], color="green", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Number of Shops")
    plt.title("Number of Shops", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    save_file = "number_of_shops.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)
    plt.show()

    # Plot Seasonality
    plt.plot(df["date"], df["seasonality"], color="orange", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Seasonality")
    plt.title("Seasonality", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    save_file = "seasonality.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)
    plt.show()

    # Plot Covid Strength
    plt.plot(df["date"], df["covid_strength"], color="red", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Covid Strength")
    plt.title("Covid Strength", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    save_file = "covid_strength.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)
    plt.show()

    # Plot Mean Suggested Price
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["mean_suggested_price"], color="purple", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Mean Suggested Price")
    plt.title("Mean Suggested Price", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    save_file = "mean_price.png"
    if output_dir:
        save_file = os.path.join(output_dir, save_file)
    plt.savefig(save_file, dpi=300)
    plt.show()


outputPath = join(join(dirname(getcwd()), "data"), "output")
dataPath = join(join(join(dirname(getcwd()), "data"), "eki"), "mmm_challenge_like.csv")
# plot_all_learners_combined(outputPath, column_name = 'T0T1')
# plot_all_learners_separate(outputPath, column_name = 'T1T2')
# plot_all_treatments(outputPath + "\\covid_sliced"
# plot_all_learners_combined(outputPath + "\\year_sliced", column_name = 'T0T1')
# plot_all_learners_combined(outputPath + "\\year_sliced\\noisy_revenue", column_name = 'T0T1')
# describe_data(dataPath, outputPath)
import shutil


def separate_folders(outputPath):
    """The function `separate_folders` takes an `outputPath` as input and moves all CSV files in that
    directory to separate folders based on the year mentioned in the file name.

    Parameters
    ----------
    outputPath
        The `outputPath` parameter is the path to the directory where the files are located.

    """
    for file in os.listdir(outputPath):
        if not file.endswith(".csv"):
            continue
        print(outputPath + "\\" + file)
        if "2020" in file:  # it will move all files with 2020 in the name
            # WARNING : If the function is run twice, it will move the files again 
            if not os.path.exists(outputPath + "_2020"):
                os.mkdir(outputPath + "_2020")
            shutil.move(outputPath + "\\" + file, outputPath + "_2020\\" + file)
        if "2021" in file:
            # WARNING : If the function is run twice, it will move the files again 
            if not os.path.exists(outputPath + "_2021"):
                os.mkdir(outputPath + "_2021")
            shutil.move(outputPath + "\\" + file, outputPath + "_2021\\" + file)


if __name__ == "__main__":
    # this is for individual testing
    myPath = outputPath + "\\year_sliced" + "\\noisy_revenue_2020"
    plot_all_learners_combined(myPath, column_name="T0T1")
    # this is for testing all possible combinations
    if 1 == 2:
        for folder in os.listdir(outputPath + "\\year_sliced"):
            if folder.endswith((".csv", ".png")):
                continue
            print(outputPath + "\\year_sliced\\" + folder)
            if folder in ["noisy_revenue", "clean_revenue"]:
                # separate_folders(outputPath + "\\year_sliced\\" + folder)
                plot_all_learners_separate(
                    outputPath + "\\year_sliced\\" + folder, column_name="T0T1"
                )
            for subfolder in os.listdir(outputPath + "\\year_sliced" + "\\" + folder):
                if subfolder.endswith((".csv", ".png")):
                    continue
                print(outputPath + "\\year_sliced\\" + folder + "\\" + subfolder)
                # separate_folders(outputPath + "\\year_sliced\\" + folder + "\\" + subfolder)
                plot_all_learners_separate(
                    outputPath + "\\year_sliced\\" + folder + "\\" + subfolder,
                    column_name="T0T1",
                )
