import matplotlib.pyplot as plt
import seaborn as sns

TARGET = "mental_state"

def basic_info(df):
    print("\n[INFO] SHAPE:", df.shape)
    print(df.info())
    print(df.describe())


def plot_distributions(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(col)
        plt.show()


def correlation_matrix(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def target_analysis(df):
    if TARGET in df.columns:
        plt.figure()
        sns.countplot(x=TARGET, data=df)
        plt.title("Mental State Distribution")
        plt.show()