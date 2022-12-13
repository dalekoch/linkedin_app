#### LOCAL APP
#### LinkedIn User Prediction App


#### Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#### Add header to describe app
st.markdown("# Programming II - Final Project")
st.markdown("#### Dale Koch")


#### Load Dataframe
s = pd.read_csv("social_media_usage.csv")


#### Create transformation function
def clean_sm(x):
    y = np.where(x==1, 1, 0)
    return y


#### Clean data
ss = pd.DataFrame({
    "income":np.where(s['income'] <= 9, s['income'], np.nan),
    "education":np.where(s['educ2'] <= 8, s['educ2'], np.nan),
    "parent":clean_sm(s['par']),
    "married":clean_sm(s['marital']),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s['age'] <= 98, s['age'], np.nan),
    "sm_li":clean_sm(s['web1h'])
})


#### Drop missing values
ss = ss.dropna()


#### Convert float to int
ss = ss.astype({"income":"int", "education":"int", "age":"int"})


#### Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]


#### Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=123) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


#### Instantiate logistic regression model
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)


#### Apply the model to the test data (predictions)
y_pred = lr.predict(X_test)


#### User inputs

# #INCOME
# income_slider = st.slider(label="Income Level", min_value=1, max_value=9)

# #EDUCATION
# education_slider = st.slider(label="Education Level", min_value=1, max_value=8)

# #PARENT
# parent_toggle = st.slider(label="Parent", min_value=0, max_value=1, help="1=Parent, 0=Not a Parent")

# #MARRIED
# married_toggle = st.slider(label="Married", min_value=0, max_value=1, help="1=Married, 0=Not Married")

# #FEMALE
# female_toggle = st.slider(label="Female", min_value=0, max_value=1, help="1=Female, 0=Not Female")

# #AGE
# age_toggle = st.slider(label="Age", min_value=1, max_value=98)

with st.sidebar:
    income_sb = st.number_input("Income (low=1 to high=9)", 1, 9)
    education_sb = st.number_input("Education (low=0 to high=8)", 0, 8)
    parent_sb = st.number_input("Parent (0=no, 1=yes)", 0, 1)
    married_sb = st.number_input("Married (0=no, 1=yes)", 0, 1)
    female_sb = st.number_input("Female (0=no, 1=yes)", 0, 1)
    age_sb = st.number_input("Age (1 to 98)", 0, 98)


#### Prediction output
person = [income_sb, education_sb, parent_sb, married_sb, female_sb, age_sb]


#### Predict class given input features
predicted_class = lr.predict([person])

# Create label for predicted class
if predicted_class == 1:
    pred_label = "LinkedIn User"
else:
    pred_label = "Not a LinkedIn User"

#### Probability of positive class (LinkedIn user = 1)
probs = lr.predict_proba([person])


#### Print predicted class and probability
# st.write(f"Predicted class: {predicted_class[0]}") # 0 = not a LinkedIn user, 1 = LinkedIn user
# st.write(f"Probability that this person is a LinkedIn user: {probs[0][1]:.2f}")

st.write("Predicted class: ", pred_label)
st.write("Probability this person is a LinkedIn user: ", probs[0][1])
