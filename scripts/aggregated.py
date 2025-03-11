import pandas as pd
import pickle
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats
import re
import os
import argparse
import warnings
import math


def load_human_data(human_aggregated_path, dataset):
    """
    Load Glasgow and Lancaster CSVs, then return dictionaries
    with relevant fields (arousal, valence, imageability, etc.).

    :param human_aggregated_path: str; Full path to the human aggregated CSV file.
    :param dataset: str; Which dataset to load. Must be either "glasgow" or "lancaster".
    :return: A dictionary of dictionaries, each containing {word -> rating}.
    """
    df_human = pd.read_csv(human_aggregated_path)

    # Glasgow
    if dataset.lower() == 'glasgow':
        #print(df_human.columns)
        human_arousal = dict(zip(df_human['Words'].str.lower(), df_human['AROU_M']))
        human_valence = dict(zip(df_human['Words'].str.lower(), df_human['VAL_M']))
        human_dominance = dict(zip(df_human['Words'].str.lower(), df_human['DOM_M']))
        human_concreteness = dict(zip(df_human['Words'].str.lower(), df_human['CNC_M']))
        human_imageability = dict(zip(df_human['Words'].str.lower(), df_human['IMAG_M']))
        human_size = dict(zip(df_human['Words'].str.lower(), df_human['SIZE_M']))
        human_gender = dict(zip(df_human['Words'].str.lower(), df_human['GEND_M']))

        return {
            "arousal":       human_arousal,
            "valence":       human_valence,
            "dominance":     human_dominance,
            "concreteness":  human_concreteness,
            "imageability":  human_imageability,
            "size":          human_size,
            "gender":        human_gender
        }

    elif dataset.lower() == 'lancaster':
        # Lancaster
        human_auditory = dict(zip(df_human['Word'].str.lower(), df_human['Auditory.mean']))
        human_gustatory = dict(zip(df_human['Word'].str.lower(), df_human['Gustatory.mean']))
        human_haptic = dict(zip(df_human['Word'].str.lower(), df_human['Haptic.mean']))
        human_interoceptive = dict(zip(df_human['Word'].str.lower(), df_human['Interoceptive.mean']))
        human_olfactory = dict(zip(df_human['Word'].str.lower(), df_human['Olfactory.mean']))
        human_visual = dict(zip(df_human['Word'].str.lower(), df_human['Visual.mean']))

        human_foot = dict(zip(df_human['Word'].str.lower(), df_human['Foot/leg.mean']))
        human_hand = dict(zip(df_human['Word'].str.lower(), df_human['Hand/arm.mean']))
        human_head = dict(zip(df_human['Word'].str.lower(), df_human['Head.mean']))
        human_mouth = dict(zip(df_human['Word'].str.lower(), df_human['Mouth/throat.mean']))
        human_torso = dict(zip(df_human['Word'].str.lower(), df_human['Torso.mean']))

        return {
            "auditory":       human_auditory,
            "gustatory":      human_gustatory,
            "haptic":         human_haptic,
            "interoceptive":  human_interoceptive,
            "olfactory":      human_olfactory,
            "visual":         human_visual,
            "foot":           human_foot,
            "hand":           human_hand,
            "head":           human_head,
            "mouth":          human_mouth,
            "torso":          human_torso
        }

    else:
        raise ValueError(
            f"Invalid dataset: '{dataset}'. Must be either 'glasgow' or 'lancaster'."
        )

def preprocessing_model_file(model_path):
    """
    Validates the CSV file name and checks for at least one '_mean' column,
    then extracts and returns the dataset name and model name from the file name.
    The file must follow the pattern '{dataset}_{model}.csv', e.g.,
    'glasgow_gpt4.csv' or 'lancaster_gpt3.csv'.

    Parameters
    ----------
    model_path : str
        Full path to the model CSV file.

    Returns
    -------
    tuple
        A tuple (dataset_name, model_name) extracted from the CSV file name.

    Raises
    ------
    ValueError
        If the file name is invalid or the CSV does not contain any columns
        ending with '_mean'.
    """
    # 1) Validate filename pattern and capture dataset, model names
    filename = os.path.basename(model_path)
    #print("filename: ", filename)
    pattern = r'^(glasgow|lancaster)_([\w\d]+)\.csv$'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(
            "Invalid file name. Must match '{dataset}_{model}.csv' "
            "e.g. 'glasgow_gpt4.csv' or 'lancaster_gpt3.csv'."
        )
    dataset_name, model_name = match.group(1), match.group(2)

    # 2) Verify the CSV has at least one '_mean' column
    df = pd.read_csv(model_path)
    if not any(col.endswith('_mean') for col in df.columns):
        raise ValueError("The CSV must contain at least one column ending with '_mean'.")

    # Return the dataset and model names if all checks pass
    return dataset_name, model_name, df



def CI(list1, list2):
    """
    Computes the 95% confidence interval for the Spearman correlation
    between two lists using bootstrap resampling.

    Parameters:
    ----------
    list1 : list or array-like
        First list of numerical values.
    list2 : list or array-like
        Second list of numerical values.

    Returns:
    -------
    tuple
        A tuple containing the lower and upper bounds of the 95% confidence interval.
    """
    # Define the number of bootstrap iterations
    n_iterations = 1000

    # Store the bootstrap correlations
    bootstrap_correlations = []

    # Get the length of the arrays
    n = len(list1)

    for _ in range(n_iterations):
        # Bootstrap resampling the indices
        indices = resample(range(n))

        # Select the corresponding items in list2
        list1 = np.array(list1)
        list2 = np.array(list2)
        sample1 = list1[indices]
        sample2 = list2[indices]

        # Calculate and store the correlation with the original list1 and the resampled list2
        correlation, _ = spearmanr(sample1, sample2)
        bootstrap_correlations.append(correlation)

    # Calculate the 95% confidence interval
    lower = np.percentile(bootstrap_correlations, 2.5)
    upper = np.percentile(bootstrap_correlations, 97.5)

    return lower, upper


def correlation(model_path, human_aggregated_path):
    """
    Computes the Spearman correlation between model-predicted values and human ratings
    across various psychological dimensions. Works for both Glasgow and Lancaster data.

    Parameters
    ----------
    model_path : str
        Full path to a model CSV file.
        The file name must be dataset_model, such as 'glagow_gpt4.csv' and 'glasgow_gpt3.csv'
        Must include columns like '{Dimension}_mean'.

    human_aggregated_path: str
        Full path to the human-aggregated CSV file (for use in load_human_data).

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the correlation results for
        a specific dimension.
    """
    # ------------------------------------------------------------------
    # 1) load human and model data
    # ------------------------------------------------------------------

    dataset, model_name, model = preprocessing_model_file(model_path)

    hd = load_human_data(human_aggregated_path, dataset)

    # define dimension_dict dynamically based on the dataset.
    # The tuple format is (DimensionType, RatingsDictionary).
    if dataset.lower() == 'glasgow':
        dimension_dict = {
            'Arousal':       ('Non-sensorimotor', hd["arousal"]),
            'Valence':       ('Non-sensorimotor', hd["valence"]),
            'Dominance':     ('Non-sensorimotor', hd["dominance"]),
            'Concreteness':  ('Non-sensorimotor', hd["concreteness"]),
            'Imageability':  ('Non-sensorimotor', hd["imageability"]),
            'Size':          ('Non-sensorimotor', hd["size"]),
            'Gender':        ('Non-sensorimotor', hd["gender"])
        }
        dataset_label = 'Glasgow'

    elif dataset.lower() == 'lancaster':
        dimension_dict = {
            'Haptic':         ('Sensory', hd["haptic"]),
            'Auditory':       ('Sensory', hd["auditory"]),
            'Olfactory':      ('Sensory', hd["olfactory"]),
            'Interoceptive':  ('Sensory', hd["interoceptive"]),
            'Visual':         ('Sensory', hd["visual"]),
            'Gustatory':      ('Sensory', hd["gustatory"]),
            'Foot/leg':       ('Motor',   hd["foot"]),
            'Hand/arm':       ('Motor',   hd["hand"]),
            'Mouth/throat':   ('Motor',   hd["mouth"]),
            'Torso':          ('Motor',   hd["torso"]),
            'Head':           ('Motor',   hd["head"])
        }
        dataset_label = 'Lancaster'
    else:
        raise ValueError(
            f"Invalid dataset: {dataset}. Must be either 'glasgow' or 'lancaster'."
        )

    # -------------------------------------------------
    # 2) Iterate over each dimension and compute the correlation
    # -------------------------------------------------
    data = []

    for dimension, (dim_type, human_dict) in dimension_dict.items():
        print(f'**{dimension}**')

        model_values = []
        human_values = []

        common_words = list(set(model['Word'])&set(list(human_dict.keys())))


        for word in common_words:
            column_name = f"{dimension}_mean"
            # Check for NaN in the value of interest
            if pd.isna(model.loc[model['Word'] == word, column_name].iloc[0]):
                warnings.warn(
                    f"There are NaN values in the model data: {model_path}",
                    UserWarning
                )
                continue

            # Get predicted value from the model
            model_val = float(model.loc[model['Word'] == word, column_name].iloc[0])
            model_values.append(model_val)

            # Get corresponding human rating
            # (Make sure the word is in human_dict; otherwise handle missing)
            human_values.append(human_dict[word])

        print(f'# of words analyzed in {dimension}: {len(model_values)}')
        # Spearman correlation
        statistic, p_value = spearmanr(model_values, human_values)

        # Compute Confidence Interval (assuming you have a CI function)
        lower_ci, upper_ci = CI(model_values, human_values)


        # Collect results
        data.append({
            'Dataset':     dataset_label,
            'Type':        dim_type,
            'Model':       model_name,
            'Dimension':   dimension,
            'Correlation': statistic,
            'N':           len(model_values),
            'CI_lower':    lower_ci,
            'CI_upper':    upper_ci,
            'Sig':         p_value
        })


    return data


def visualize_results(df, output_path):
    """
    Reads a CSV file that contains at least the columns:
      - 'Model'
      - 'Dimension'
      - 'Correlation'
      - 'CI_Lower'
      - 'CI_Upper'

    Then:
      1) Groups data by Model
      2) Creates a grid of subplots with 2 plots per row (e.g., if you have 3 models, it’ll produce 2 rows x 2 cols,
         with the last subplot possibly empty if the number of models is odd).
      3) In each subplot, it shows a bar chart of 'Correlation' by 'Dimension' with error bars for the CI.

    Parameters
    ----------
    df: pd.DataFrame
    output_path : str
        Path for saving the output (with .csv as the extension). The output figure will be named following the csv file
    """

    file_without_ext, file_extension = os.path.splitext(output_path)
    figure_output_path = f"{file_without_ext}.png"

    # Check columns
    required_cols = {'Model', 'Dimension', 'Correlation', 'CI_lower', 'CI_upper'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"The dataframe must contain at least these columns: {required_cols}")

    # Get unique models, sorted for consistent ordering
    models = sorted(df['Model'].unique())
    n_models = len(models)

    # Determine rows and columns (2 columns per row)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    # Create figure and axes array
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    axes = axes.flatten()  # Flatten to 1D for easier indexing

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Subset data for this model
        sub_df = df[df['Model'] == model].copy()
        #sub_df.sort_values('Dimension', inplace=True)

        # Prepare x-axis positions
        x_positions = np.arange(len(sub_df))

        # Extract correlation values
        correlations = sub_df['Correlation'].values

        # Compute error (distance from correlation to lower/upper bounds)
        ci_lower = sub_df['CI_lower'].values
        ci_upper = sub_df['CI_upper'].values

        y_err_lower = correlations - ci_lower
        y_err_upper = ci_upper - correlations

        # Plot bars with error bars
        ax.bar(x_positions, correlations, yerr=[y_err_lower, y_err_upper], capsize=4)

        # Label ticks with dimension names
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sub_df['Dimension'], rotation=45, ha='right')

        # Title, labels, reference line
        ax.set_title(f"Model: {model}")
        ax.set_ylabel("Correlation")
        ax.axhline(0, linestyle='--', linewidth=1, color='gray')
        ax.set_ylim(-0.1, 1)

    # If there are unused subplots (e.g., odd # of models), hide them
    for i in range(n_models, n_rows*n_cols):
        fig.delaxes(axes[i])

    plt.grid(False)
    plt.tight_layout()

    plt.savefig(figure_output_path, dpi=100)


def list_valid_files(directory):
    """
    Returns a list of files in `directory`, ignoring known system files.
    Works for both macOS and Windows platforms.
    """
    # Common system files to exclude
    excluded_files = {'.DS_Store', 'Thumbs.db', 'desktop.ini'}

    # Get files in directory, filtering out system files and non-files (subdirectories)
    valid_files = [
        f for f in os.listdir(directory)
        if f not in excluded_files and os.path.isfile(os.path.join(directory, f))
    ]
    return valid_files

def main_analyses(human_aggregated_directory: str,
                  model_directory: str,
                  output_path: str):
    """
    Conduct main analyses by iterating over model files in `model_directory` and
    human-aggregated data files in `human_aggregated_directory`. For each
    combination, a correlation function is called and its list of dictionaries
    is collected. Finally, all dictionaries are converted into a DataFrame and
    saved to the specified `output_path`.

    :param human_aggregated_directory: Path to directory containing 1 or 2 human
                                       aggregated files. If `dataset` != 'both',
                                       it must contain exactly 1 file. If
                                       `dataset` == 'both', it must contain 2.
    :param dataset: One of {'glasgow', 'lancaster', 'both'}, specifying which
                    dataset(s) is/are being analyzed.
    :param model_directory: Directory containing valid model files to be analyzed.
    :param output_path: Filepath where the final compiled DataFrame should be saved.
    """

    # Helper: ensure the user-supplied directory actually exists
    if not os.path.isdir(human_aggregated_directory):
        raise NotADirectoryError(
            f"human_aggregated_directory is not a valid directory: {human_aggregated_directory}"
        )

    if not os.path.isdir(model_directory):
        raise NotADirectoryError(
            f"model_directory is not a valid directory: {model_directory}"
        )


    # Accumulate all results here
    all_results = []

    # Iterate over each model file in model_directory
    print('files to be processed in model_directory: ', list_valid_files(model_directory))
    for model_file in sorted(list_valid_files(model_directory)):
        model_path = os.path.join(model_directory, model_file)
        print(f'---------------processing {model_path}---------------')
        dataset, _, _ = preprocessing_model_file(model_path)

        # Skip directories (or handle them differently if needed)
        if not os.path.isfile(model_path):
            continue

        human_file_path = os.path.join(human_aggregated_directory, f"{dataset}_human.csv")

        # Check if the file exists
        if not os.path.isfile(human_file_path):
            raise ValueError(
                f"Expected '{dataset}_human.csv' file in {human_aggregated_directory} "
                f"but couldn't find it."
            )

        correlation_data = correlation(model_path, human_file_path)
        # Accumulate
        all_results.extend(correlation_data)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV (or any other format you prefer)
    df.to_csv(output_path, index=False)

    # Visualization
    visualize_results(df, output_path)

    print(f"Analysis complete. Results saved to {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run main analyses.")
    parser.add_argument("--human_aggregated_directory", type=str, required=True,
                        help="Path to directory containing human-aggregated data files.")
    parser.add_argument("--model_directory", type=str, required=True,
                        help="Path to directory containing valid model files.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the final analysis results in csv format.")

    args = parser.parse_args()

    main_analyses(
        human_aggregated_directory=args.human_aggregated_directory,
        model_directory=args.model_directory,
        output_path=args.output_path
    )
