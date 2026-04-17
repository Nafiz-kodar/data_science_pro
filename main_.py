from src.Data_injestion import load_data, save_data
from src.Cleaning import clean_data
from src.Pre_process import preprocess_data
from src.eda import basic_info, plot_distributions, correlation_matrix, target_analysis

CLEAN_PATH = "data/processed/cleaned_data.csv"

def main():

    # STEP 1: LOAD
    df = load_data()

    # STEP 2: CLEAN
    df_clean = clean_data(df)

    # SAVE CLEANED DATA )
    save_data(df_clean, CLEAN_PATH)

    # STEP 3: EDA
    basic_info(df_clean)
    plot_distributions(df_clean)
    correlation_matrix(df_clean)
    target_analysis(df_clean)

    # STEP 4: PREPROCESS
    X, preprocessor = preprocess_data(df_clean)

    print("\n[INFO] PIPELINE COMPLETE")


if __name__ == "__main__":
    main()