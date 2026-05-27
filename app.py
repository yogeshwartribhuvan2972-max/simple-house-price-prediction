import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Title
st.title("House Price Prediction App")

# Load Dataset
df = pd.read_csv("house_prices_multivariate.csv.xls")

# Show Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Heatmap
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,7))

corr = df[["Bedrooms", "Size_sqft", "Age_years", "Price_Lakhs_INR"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="YlGnBu",
    linewidths=0.5,
    fmt='.2f',
    ax=ax
)

st.pyplot(fig)

# Features & Target
X = df.iloc[:,0:3]
Y = df.iloc[:,-1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=6
)

# Model Training
lr = LinearRegression()
lr.fit(X_train, y_train)

# Accuracy
st.subheader("Model Accuracy")

train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)

st.write("Training Score:", train_score)
st.write("Testing Score:", test_score)

# User Input
st.subheader("Predict House Price")

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

size = st.number_input("Size in Sqft", min_value=300, max_value=10000, value=1500)

age = st.number_input("Age of House", min_value=0, max_value=100, value=10)

# Prediction
if st.button("Predict Price"):

    prediction = lr.predict([[bedrooms, size, age]])

    st.success(f"Predicted House Price: {prediction[0]:.2f} Lakhs INR")
