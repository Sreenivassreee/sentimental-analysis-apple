from coremltools import converters
from coremltools.models import MLModel
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV data
data = pd.read_csv("twitter-sanders-apple3.csv")

# Split the data into training and testing sets
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=5)

# Create an MLTextClassifier
sent_classifier = converters.sklearn.convert(training_data, target="class", features=["text"])

# Evaluate the model
evaluation_metrics = sent_classifier.evaluate(testing_data, target="class", features=["text"])
accuracy = (1.0 - evaluation_metrics["classification_error"]) * 100

# Save the model
metadata = {
    "author": "Sreenivas",
    "short_description": "This is trained sentiment classifier",
    "version": "1.0"
}
mlmodel = MLModel(sent_classifier, metadata=metadata)
mlmodel.save("TweetClassifier.mlmodel")

# Make predictions
prediction1 = mlmodel.predict({"text": "@apple is a waste company"})
prediction2 = mlmodel.predict({"text": "@apple is love"})
