import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset.csv")
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("="*50)
print(" PHISHING EMAIL DETECTION MODEL ")
print("="*50)

print(f"\nModel Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, prediction))

while True:

    print("\nEnter Email Text")
    email = input(">> ")

    email_vector = vectorizer.transform([email])

    result = model.predict(email_vector)

    print("\nPrediction :", result[0].upper())

    choice = input("\nCheck another email? (yes/no): ")

    if choice.lower() != "yes":
        print("\nThank You!")
        break