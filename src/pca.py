import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pd.DataFrame(X_train_scaled).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("X_test_scaled.csv", index=False)

print("Scaling done.")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nPCA done.")
print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)
print("Number of PCA components selected:", pca.n_components_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]

X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)

X_train_pca_df.to_csv("X_train_pca.csv", index=False)
X_test_pca_df.to_csv("X_test_pca.csv", index=False)

# Optional explained variance report
explained_variance_df = pd.DataFrame({
    "principal_component": pca_columns,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": pca.explained_variance_ratio_.cumsum()
})
explained_variance_df.to_csv("pca_explained_variance.csv", index=False)

print("\nSaved successfully:")
print("- X_train_pca.csv")
print("- X_test_pca.csv")
print("- pca_explained_variance.csv")
