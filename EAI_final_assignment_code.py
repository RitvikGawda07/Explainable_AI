import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_text import LimeTextExplainer 
import matplotlib.pyplot as plt
import re
#===================== Training the a Logestic Regression with the Dataset ============================


# Loading the Dataset and using only the relevent columns from the original Dataset
data = pd.read_csv("youtoxic_english_1000.csv")
data = data[['Text','IsToxic']]
data = data.rename(columns={'Text':'text','IsToxic':'label'})


# Initializing the X and Y variables 
X = data['text']
y = data['label'] 


# Splitting the Data into Train Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)  


# Coverting the comments (Texts) into words so that the model will be able to make predictions in a better way. Each word in the comment gets converted a vector with numbers
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000)   


# Transforming the Comments (Texts) to numbers 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 


# Initializing the Model and fitting the training set into the model
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec) 


# Results from the model
print("Accuracy of the Classifcation model: ", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))


#===================== LIME Explanation Part ============================


# Create LIME explainer for text classification
explainer = LimeTextExplainer(class_names=['non-toxic','toxic'])


# convert text to TF-IDF vectors and return probability for each class
def pred_proba(texts):
    texts_vec = vectorizer.transform(texts)
    return model.predict_proba(texts_vec)

sample_text = X_test.iloc[14]
print('sample text: ', sample_text)


# Generate LIME explanation for the selected comment
exp = explainer.explain_instance(
    sample_text,
    pred_proba,
    num_features=10)


# Convert LIME output into a table format
def lime_to_table(lime_explanations):

     # create table
    df = pd.DataFrame(lime_explanations, columns=["Word", "Weight"])
    df["Importance"] = df["Weight"].abs()
    # sort by contribution
    df = df.sort_values(by="Weight", ascending=False) 
    return df

# Plotting graph showing the weights for each word in a sentence 
def plot_lime_explanations(lime_explanations):
    df = pd.DataFrame(lime_explanations, columns=["Word", "Weight"])
    # sort for horizontal plot
    df = df.sort_values(by="Weight", ascending=True) 

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Word"], df["Weight"])  

    plt.xlabel("LIME Weight")
    plt.ylabel("Word")
    plt.title("LIME Explanation for Sentence")

    # Add weight values next to bars
    for i, weight in enumerate(df["Weight"]):
        plt.text(weight, i, f"{weight:.3f}", va="center")

    plt.tight_layout()
    plt.show()

print("LIME explanations:")
print(lime_to_table(exp.as_list()))
print()
plot_lime_explanations(exp.as_list())

#===================== Counterfactual Part ============================

# Clean words by removing special characters and converting to lowercase
def clean_word(word):
    return re.sub(r"[^a-zA-Z0-9]", "", word.lower())


# Generate simple counterfactual explanation
def simple_counterfactual(original_text, model, vectorizer, explainer):


     # Check original prediction
    original_pred = model.predict(vectorizer.transform([original_text]))[0] 
    original_prob = model.predict_proba(vectorizer.transform([original_text]))[0][1]

    print("Original prediction:", original_pred)
    print("Original toxic probability:", round(original_prob, 4))

   
    if original_pred == 0:  # If already non-toxic, return it as is
        return original_text

    # Get LIME explanation
    exp = explainer.explain_instance(
        original_text,
        pred_proba,
        num_features=10)
 
    # Select only words contributing to toxic class
    toxic_words = [(word, weight) for word, weight in exp.as_list() if weight > 0] 
    toxic_words = sorted(toxic_words, key=lambda x: x[1], reverse=True)

    print("Toxic words from LIME:", toxic_words)

    words = original_text.split()
    current_text = original_text

    # Remove words one by one from most toxic to least toxic
    for toxic_word, weight in toxic_words:
        cleaned_toxic = clean_word(toxic_word)

        new_words = []
        removed = False
 
        # Remove first occurrence of the toxic word
        for w in words:
            if not removed and clean_word(w) == cleaned_toxic:
                removed = True
                continue
            new_words.append(w)

        words = new_words
        current_text = " ".join(words)

        # Get new prediction after removing the word
        new_pred = model.predict(vectorizer.transform([current_text]))[0]
        new_prob = model.predict_proba(vectorizer.transform([current_text]))[0][1]

        print(f"\nRemoved word: {toxic_word}")
        print("New sentence:", current_text)
        print("New prediction:", new_pred)
        print("New toxic probability:", round(new_prob, 4))

        # If prediction flips to non-toxic, return result
        if new_pred == 0:
            return current_text

    return current_text

example = X_test.iloc[83]

# Generate counterfactual explanation
cf = simple_counterfactual(example, model, vectorizer, explainer)
print("\nOriginal:")
print(example)
print("\nCounterfactual sentence:")
print(cf)

