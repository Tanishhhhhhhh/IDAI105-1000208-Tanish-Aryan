# Amazon Market Basket Analysis

This project performs **market basket analysis** on an Amazon dataset to uncover insights into customer behavior, product ratings, and pricing trends. The analysis includes data cleaning, exploratory data analysis (EDA), customer segmentation, association rule mining, and user behavior analysis.

## Project Overview

The goal of this project is to analyze an Amazon dataset to:
- Clean and preprocess the data.
- Perform exploratory data analysis (EDA) to understand the distribution of prices, ratings, and categories.
- Segment customers using clustering techniques.
- Discover association rules between products using the Apriori algorithm.
- Analyze user behavior and create customer profiles.

---

## Features

- **Data Cleaning and Preprocessing**:
  - Handles missing values in critical columns.
  - Cleans and normalizes numerical data (e.g., prices, ratings).
  - Encodes categorical variables (e.g., product categories).

- **Exploratory Data Analysis (EDA)**:
  - Visualizes the distribution of discounted and actual prices using histograms and box plots.
  - Examines relationships between variables using scatter plots.
  - Analyzes product ratings and category distributions using bar charts.
  - Identifies correlations between numerical features using a heatmap.

- **Customer Segmentation**:
  - Uses K-Means clustering to segment customers based on their purchasing behavior.
  - Visualizes customer segments using scatter plots.

- **Association Rule Mining**:
  - Implements the Apriori algorithm to discover frequent itemsets.
  - Calculates support, confidence, and lift for association rules.

- **User Behavior Analysis**:
  - Creates customer profiles based on ratings, prices, and review counts.
  - Analyzes review data to identify trends.

---

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `mlxtend`, `streamlit`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/amazon-market-basket-analysis.git
   cd amazon-market-basket-analysis
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (`amazon.csv`) and place it in the project directory.

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run Amazon.py
   ```

2. Open the app in your browser:
   - The app will display EDA visualizations, customer segmentation results, association rules, and user behavior analysis.
---

## Results

### EDA Visualizations
- **Histogram of Discounted Price**: Shows the distribution of discounted prices.
- **Box Plot of Actual Price**: Displays the spread and outliers of actual prices.
- **Scatter Plot of Actual Price vs Discounted Price**: Examines the relationship between actual and discounted prices.
- **Bar Chart of Product Ratings**: Visualizes the distribution of product ratings.
- **Heatmap of Correlations**: Identifies correlations between numerical features.

### Customer Segmentation
- **Scatter Plot of Customer Segments**: Visualizes customer clusters based on purchasing behavior.

### Association Rule Mining
- **Association Rules**: Displays rules with support, confidence, and lift metrics.

### User Behavior Analysis
- **Customer Profiles**: Summarizes customer behavior based on ratings, prices, and review counts.
- **Review Analysis**: Analyzes review data to identify trends.

---
### Python File link
https://drive.google.com/file/d/1D7IQAd7wM_HhRUuldmaofza3VT5-agS-/view?usp=sharing
### Dataset Link
https://drive.google.com/file/d/1x4lc25yo5AzM5HDMQpkUiC5l7M4wARWT/view?usp=sharing
