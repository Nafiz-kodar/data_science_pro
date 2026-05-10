import pandas as pd
import numpy as np


df = pd.read_csv("cleaned_data.csv")

print("Original shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


def get_season(month):
    if pd.isna(month):
        return np.nan
    elif month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


df["season"] = df["month"].apply(get_season)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 24, 34, 49, 100],
    labels=["<25", "25-34", "35-49", "50+"]
)

df["social_media_ratio"] = df["social_media_time_min"] / (df["daily_screen_time_min"] + 1)
df["neg_pos_ratio"] = df["negative_interactions_count"] / (df["positive_interactions_count"] + 1)
df["activity_sleep_ratio"] = df["physical_activity_min"] / (df["sleep_hours"] + 1)
df["screen_sleep_ratio"] = df["daily_screen_time_min"] / (df["sleep_hours"] + 1)

df["digital_overload"] = df["daily_screen_time_min"] - df["physical_activity_min"]
df["screen_minus_social"] = df["daily_screen_time_min"] - df["social_media_time_min"]
df["interaction_balance"] = df["positive_interactions_count"] - df["negative_interactions_count"]
df["wellbeing_gap"] = (
    df["physical_activity_min"] + (df["sleep_hours"] * 60)
    - df["daily_screen_time_min"]
)

df["screen_x_sleep"] = df["daily_screen_time_min"] * df["sleep_hours"]
df["social_x_negative"] = df["social_media_time_min"] * df["negative_interactions_count"]
df["activity_x_positive"] = df["physical_activity_min"] * df["positive_interactions_count"]
df["stress_x_anxiety"] = df["stress_level"] * df["anxiety_level"]
df["mood_x_sleep"] = df["mood_level"] * df["sleep_hours"]

df["high_screen_time"] = (df["daily_screen_time_min"] > 300).astype(int)
df["low_sleep"] = (df["sleep_hours"] < 6).astype(int)
df["low_activity"] = (df["physical_activity_min"] < 30).astype(int)
df["high_negative_interaction"] = (df["negative_interactions_count"] > 10).astype(int)

df = df.drop(columns=["person_name", "date"])

categorical_cols = ["gender", "platform", "season", "age_group"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nFeature engineered shape:", df.shape)
print("\nFeature engineered columns:")
print(df.columns.tolist())

df.to_csv("feature_engineered_data.csv", index=False)
print(f"\nSaved successfully as 'feature_engineered_data.csv'")
