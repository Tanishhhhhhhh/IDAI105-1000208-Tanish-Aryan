import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

# Load the dataset
data = pd.read_csv('amazon.csv')

# Print the columns in the DataFrame
print("Columns in DataFrame:", data.columns.tolist())

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check for missing values in all critical columns
critical_columns = ['product_id', 'actual_price', 'discounted_price', 'rating', 'rating_count', 'user_id']
print("Missing values in critical columns:")
print(data[critical_columns].isnull().sum())

# Drop rows with missing values in critical columns
data.dropna(subset=critical_columns, inplace=True)

# Clean 'actual_price' and 'discounted_price' columns
def clean_price_column(column):
    return column.replace({'₹': '', ',': '', '|': '', ' ': ''}, regex=True).astype(float)

data['actual_price'] = clean_price_column(data['actual_price'])
data['discounted_price'] = clean_price_column(data['discounted_price'])

# Ensure 'rating' and 'rating_count' are numeric and handle any non-numeric values
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data['rating_count'] = pd.to_numeric(data['rating_count'], errors='coerce')
data.dropna(subset=['rating', 'rating_count'], inplace=True)  # Drop rows where rating or rating_count could not be converted

# Normalize numerical data
scaler = StandardScaler()
data[['actual_price', 'discounted_price', 'rating', 'rating_count']] = scaler.fit_transform(data[['actual_price', 'discounted_price', 'rating', 'rating_count']])

# Check if 'category' exists before encoding
if 'category' in data.columns:
    data = pd.get_dummies(data, columns=['category'], drop_first=True)
else:
    print("Column 'category' not found in the DataFrame.")

# Streamlit App
st.title('Amazon Market Basket Analysis')

# Display EDA results
st.subheader('Exploratory Data Analysis (EDA)')

# Histogram for Discounted Price
st.subheader('Distribution of Discounted Price')
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data['discounted_price'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Box Plot for Actual Price
st.subheader('Box Plot of Actual Price')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x=data['actual_price'], ax=ax)
st.pyplot(fig)

# Scatter Plot for Actual Price vs Discounted Price
st.subheader('Scatter Plot of Actual Price vs Discounted Price')
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='actual_price', y='discounted_price', data=data, ax=ax)
st.pyplot(fig)

# Round ratings to 1 decimal place to reduce overcrowding
data['rating_rounded'] = data['rating'].round(1)

# Bar Chart for Product Rating Distribution
st.subheader('Product Rating Distribution')
fig, ax = plt.subplots(figsize=(14, 6))
sns.countplot(x='rating_rounded', data=data, ax=ax, width=0.6)
plt.xticks(rotation=45)
plt.xlabel('Rating (Rounded to 1 Decimal Place)')
plt.ylabel('Count')
st.pyplot(fig)

# Category Distribution
if 'category' in data.columns:
    st.subheader('Product Category Distribution')
    category_counts = data['category'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Heatmap for Correlation
st.subheader('Correlation Heatmap')
numeric_data = data.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Customer Segmentation using K-Means Clustering
st.subheader('Customer Segmentation Clusters')
features = data[['discounted_price', 'actual_price', 'rating']]
kmeans = KMeans(n_clusters=5, random_state=42)
data['customer_segment'] = kmeans.fit_predict(features)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='actual_price', y='discounted_price', hue='customer_segment', data=data, palette='viridis', ax=ax)
st.pyplot(fig)

# Prepare data for association rule mining
basket = (data.groupby(['user_id', 'product_id'])['rating']
          .sum().unstack().reset_index().fillna(0)
          .set_index('user_id'))

# Convert values to 1 and 0
def encode_units(x):
    return 1 if x > 0 else 0  # Consider any rating > 0 as an interaction

basket_encoded = basket.applymap(encode_units)

# Generate frequent itemsets with a lower min_support
frequent_itemsets = apriori(basket_encoded, min_support=0.0001, use_colnames=True)  # Lowered min_support

# Check if frequent_itemsets is empty
if frequent_itemsets.empty:
    st.subheader('Association Rules')
    st.write("No frequent itemsets found. This could be due to a high `min_support` threshold or insufficient data.")
    st.write("Try lowering the `min_support` value or checking the data preparation steps.")
else:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    st.subheader('Association Rules')
    st.write(rules)

# User Behavior Analysis: Create Customer Profiles
st.subheader('Customer Profiles')
customer_profiles = data.groupby('user_id').agg({
    'rating': 'mean',
    'actual_price': 'sum',
    'discounted_price': 'sum',
    'rating_count': 'sum'
}).reset_index()
st.write(customer_profiles)

# Analyze Review Data
st.subheader('Review Analysis')
review_analysis = data.groupby('product_id').agg({
    'review_content': 'count',
    'rating': 'mean'
}).reset_index()
st.write(review_analysis)
