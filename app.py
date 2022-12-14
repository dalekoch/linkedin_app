#### LinkedIn User Prediction App
#### Final Project
#### OPIM 607 Programming II
#### Dale Koch
#### 12/13/2022


###### IMPORT PACKAGES ######
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


###### HEADER ######
st.set_page_config(page_title = "LinkedIn Predictor by Dale", layout = "centered")

st.markdown("### LinkedIn User Prediction")
st.markdown("###### Created by Dale Koch")
st.markdown("---")

###### INSTRUCTIONS ######
st.write("Enter values in the boxes below to predict whether someone with those characteristics is a LinkedIn user. You may use the Tab key and enter number values with your keyboard. Press enter to select a value before tabbing to the next box.")

###### DATA & LOG REGRESSION ######

## Load Dataframe
s = pd.read_csv("social_media_usage.csv")

## Create transformation function
def clean_sm(x):
    y = np.where(x==1, 1, 0)
    return y

## Clean data
ss = pd.DataFrame({
    "income":np.where(s['income'] <= 9, s['income'], np.nan),
    "education":np.where(s['educ2'] <= 8, s['educ2'], np.nan),
    "parent":clean_sm(s['par']),
    "married":clean_sm(s['marital']),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s['age'] <= 98, s['age'], np.nan),
    "sm_li":clean_sm(s['web1h'])
})

## Drop missing values
ss = ss.dropna()

## Convert float to int
ss = ss.astype({"income":"int", "education":"int", "age":"int"})

## Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

## Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=123) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

## Instantiate logistic regression model
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

## Apply the model to the test data (predictions)
y_pred = lr.predict(X_test)


###### USER INPUT ######

## Create dictionaries
income_options = {
    1: "1 - Less than $10,000",
	2: "2 - $10,000 - $19,999",
	3: "3 - $20,000 - $29,999",
	4: "4 - $30,000 - $39,999",
	5: "5 - $40,000 - $49,999",
	6: "6 - $50,000 - $74,999",
	7: "7 - $75,000 - $99,999",
	8: "8 - $100,000 - $149,999",
	9: "9 - $150,000 or more"
}

education_options = {
    1: "1 - Less than high school (Up to grade 8)",
	2: "2 - Some high school, no diploma",
	3: "3 - High school graduate or GED",
	4: "4 - Some college (no degree)",
	5: "5 - Two-year associate degree",
	6: "6 - Bachelor’s degree (e.g., BS, BA, AB)",
	7: "7 - Some graduate school (no degree)",
	8: "8 - Postgrad degree (e.g., MA, MS, PhD, MD, JD)",
}

parent_options = {
    0: "0 - No",
    1: "1 - Yes"
}

married_options = {
    0: "0 - No",
    1: "1 - Yes"
}

female_options = {
    0: "0 - No",
    1: "1 - Yes"
}

## Input fields
col1, col2 = st.columns(2)

with col1:
    income_value = st.selectbox(label = "Income range", options = (1, 2, 3, 4, 5, 6, 7, 8, 9), format_func = lambda x: income_options.get(x))
    education_value = st.selectbox(label = "Education Level", options = (1, 2, 3, 4, 5, 6, 7, 8), format_func = lambda x: education_options.get(x))
    parent_value = st.selectbox("Parent?", options = (0, 1), format_func = lambda x: parent_options.get(x))

with col2:
    married_value = st.selectbox("Married?", options = (0, 1), format_func = lambda x: married_options.get(x))
    female_value = st.selectbox("Female?", options = (0, 1), format_func = lambda x: female_options.get(x))
    age_value = st.number_input("Age (1 to 98)", 0, 98)


###### MAKE PREDICTION ######

## Store prediction indicator values as object
person = [income_value, education_value, parent_value, married_value, female_value, age_value]

## Store predicted class as object
predicted_class = lr.predict([person])

### Create label for predicted class
if predicted_class == 1:
    pred_label = "**LinkedIn User**"
else:
    pred_label = "**Not a LinkedIn User**"

## Probability of positive class (LinkedIn user = 1)
probs = lr.predict_proba([person])

#st.markdown("---")


###### OUTPUT ######

## Store predition as string
pred_disp = ("Predicted class: " + pred_label)

## Output prediction & probability
if predicted_class == 1:
    st.success(pred_disp)
else:
    st.error(pred_disp)

st.write("**Probability this person is a LinkedIn user:** ", probs[0][1])