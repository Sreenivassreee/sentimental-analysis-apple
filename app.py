import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import coremltools
from coremltools.models import MLModel


data = pd.read_csv("twitter-sanders-apple3.csv")
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=5)



tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(data["text"])
y_train = data["class"]

model_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(multi_class='ovr'))
])

model_pipeline.fit(X_train, y_train)

model_pipeline.multi_class = 'ovr'
coreml_model = coremltools.converters.sklearn.convert(model_pipeline)

try:
    coreml_model = MLModel("TweetClassifier.mlmodel") 
    print("----------- Already saved ---------------")
except:
    coreml_model.save("TweetClassifier.mlmodel")
    print("----------- Saved ---------------")





for i in range(0,100):
    input_text =input("Sentence : ")
    input_features = tfidf_vectorizer.transform([str(input_text)])
    input_features_dense = input_features.toarray()
    prediction = coreml_model.predict({"input": input_features_dense})
    print(prediction)


