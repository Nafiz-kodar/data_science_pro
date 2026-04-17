from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET = "mental_state"

def preprocess_data(df):

    # remove target from features
    X = df.drop(columns=[TARGET], errors="ignore")

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    print(f"[INFO] Preprocessed shape: {X_processed.shape}")

    return X_processed, preprocessor