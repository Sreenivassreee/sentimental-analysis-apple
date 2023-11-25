import pandas as pd
from sklearn.datasets import make_classification

# Generate a random dataset
data = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
df = pd.DataFrame(data[0], columns=["feature1", "feature2"])
df["class"] = data[1]

# Save the dataset to a CSV file
csv_path = "twitter-sanders-apple3.csv"
df.to_csv(csv_path, index=False)

# Now you can use the provided Python code with this CSV file
