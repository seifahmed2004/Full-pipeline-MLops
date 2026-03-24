from sklearn.datasets import load_iris
import pandas as pd
import os

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Create data folder
os.makedirs("data", exist_ok=True)

# Save CSV
df.to_csv("data/iris.csv", index=False)

print("Saved to data/iris.csv")