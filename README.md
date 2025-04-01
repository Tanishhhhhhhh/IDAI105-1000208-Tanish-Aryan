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

🔍 Analysis Performed
1️⃣ Data Cleaning & Preprocessing
Handled missing values in key columns (Delivery_person_Ratings, Time_taken(min), Weatherconditions).

Removed outliers using IQR for numerical columns (Time_taken(min)).

Normalized numerical data (e.g., Delivery_person_Age, Delivery_person_Ratings).

Encoded categorical variables (Weatherconditions, Road_traffic_density, Type_of_vehicle).

📌 Key Insights:
✔️ Ensured data consistency for further analysis.
✔️ Improved model accuracy by handling missing values and outliers.

2️⃣ Exploratory Data Analysis (EDA)
📊 Distribution Analysis
Histograms & Box Plots for Time_taken(min) and Delivery_person_Ratings.

Scatter Plots to analyze relationships (e.g., Delivery Time vs. Ratings).

📈 Correlation Analysis
Heatmap to identify correlations between numerical features (Age, Ratings, Delivery Time).

📌 Key Findings
✔️ Most deliveries take 20-40 minutes.
✔️ Higher-rated delivery personnel tend to have faster delivery times.
✔️ Bad weather (rain, fog) increases delivery time.

3️⃣ Customer Segmentation (K-Means Clustering)
Clustered customers based on:

Delivery Time

Delivery_person_Ratings

Order Frequency

Visualized clusters using scatter plots.

📌 Key Insights:
✔️ Identified 3 customer segments (Fast & Reliable, Slow but High-Rated, Inconsistent).
✔️ Helps Uber Eats optimize delivery personnel allocation.

4️⃣ Association Rule Mining (Apriori Algorithm)
Discovered frequent itemsets (e.g., Snacks + Drinks are often ordered together).

Calculated metrics (Support, Confidence, Lift) to find strong associations.

📌 Key Insights:
✔️ "Meal + Drinks" has high confidence (75%).
✔️ "Festival days" lead to more Buffet orders.

5️⃣ User Behavior Analysis
Analyzed review trends (e.g., higher ratings on weekends).

Customer profiles based on:

Order frequency

Average delivery time

Preferred food categories

📌 Key Insights:
✔️ Urban customers prefer fast-food orders.
✔️ Metropolitan customers order more meals during weekdays****
---

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `mlxtend`, `streamlit`
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
### Graph Inage link
https://drive.google.com/drive/folders/1QXyuEWUlY0msmWkeqFFNiKfirqJ3TtSa?usp=sharing
