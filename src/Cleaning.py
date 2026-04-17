import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    df.columns = df.columns.str.strip().str.lower()      # normalize column names

    if "person_name" in df.columns:          # person_name cleanup
        df["person_name"] = (
            df["person_name"]
            .astype(str)
            .str.strip()
            .str.title()
        )


    if "gender" in df.columns:              # gender standardization
        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "m": "male",
                "f": "female",
                "male ": "male",
                "female ": "female"
            })
        )

    if "platform" in df.columns:        # platform normalization
        df["platform"] = (
            df["platform"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # =========================
    # 3. MISSING VALUE HANDLING
    # =========================

    numeric_cols = [                        # numeric columns → median
        "age",
        "daily_screen_time_min",
        "social_media_time_min",
        "negative_interactions_count",
        "positive_interactions_count",
        "sleep_hours",
        "physical_activity_min"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [                    # categorical columns → mode    
        "anxiety_level",
        "stress_level",
        "mood_level"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # =========================
    # 4. SPECIAL LOGIC (IMPORTANT)
    # =========================

    # If social_media_time missing → use daily_screen_time * 0.7 (assumption)
    if "social_media_time_min" in df.columns and "daily_screen_time_min" in df.columns:
        df["social_media_time_min"] = df["social_media_time_min"].fillna(
            df["daily_screen_time_min"] * 0.7
        )

    time_cols = [                                           # Ensure no negative values in time features
        "daily_screen_time_min",
        "social_media_time_min",
        "sleep_hours",
        "physical_activity_min"
    ]

    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # =========================
    # 5. TARGET SAFETY (VERY IMPORTANT)
    # =========================

    # DO NOT modify target
    # mental_state stays untouched

    print(f"[INFO] Cleaned data shape: {df.shape}")

    return df