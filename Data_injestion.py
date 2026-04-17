import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/mental_health_social_media_dataset.csv")
    print(f"[INFO] Raw data loaded: {df.shape}")
    return df