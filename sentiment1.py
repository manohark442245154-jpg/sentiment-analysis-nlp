import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading positive and negative reviews from folders
def load_data(path):
    texts = []
    labels = []
    for sentiment in ['pos', 'neg']:
        folder = os.path.join(path, sentiment)
        for filename in os.listdir(folder):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(folder, filename)
            with open(filepath, encoding='utf-8') as f:
                texts.append(f.read())
            # 1 for positive, 0 for negative
            if sentiment == 'pos':
                labels.append(1)
            else:
                labels.append(0)
    return texts, labels

# change this path if needed
train_path = r"C:\Users\manohar\Downloads\aclImdb\train"

print("loading reviews...")
texts, labels = load_data(train_path)
print("total reviews loaded:", len(texts))

# putting data into a dataframe
df = pd.DataFrame({'text': texts, 'label': labels})

# using only 5000 reviews to make it faster
df = df.sample(5000, random_state=42)

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# converting text to numbers using tfidf
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# training the model
clf = LogisticRegression(max_iter=500)
clf.fit(X_train_tfidf, y_train)

# checking accuracy
predictions = clf.predict(X_test_tfidf)
acc = accuracy_score(y_test, predictions)
print("model accuracy:", round(acc * 100, 2), "%")

print("\nsentiment analyzer is ready")
print("type a sentence to check if it is positive or negative")
print("type exit to quit\n")

# taking input from user
while True:
    sentence = input("enter sentence: ")
    if sentence.lower() == 'exit':
        print("bye!")
        break
    vec = tfidf.transform([sentence])
    result = clf.predict(vec)[0]
    if result == 1:
        print("positive sentiment\n")
    else:
        print("negative sentiment\n")