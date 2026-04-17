import pandas as pd

RAW_PATH = "data/raw/mental_health_social_media_dataset.csv"
CLEAN_DIR = "data/clean"


# =========================
# LOAD RAW DATA
# =========================
def load_data():
    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Loaded raw data: {df.shape}")
    return df


# =========================
# SAVE DATA (CLEAN / PROCESSED)
# =========================
def save_data(df, filename="cleaned_data.csv"):

    df.to_csv("data/clean/cleaned_data.csv", index=False)

    print(f"[INFO] Saved data to: {"data/clean/cleaned_data.csv"}")