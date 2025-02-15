# Customer Segmentation 

## Overview
This project performs customer segmentation using the K-Means clustering algorithm. The dataset contains customer information, including age, annual income, spending score, and purchase frequency. The goal is to segment customers based on their purchasing behavior to help businesses understand and target their audience effectively.

## Dataset
The dataset used is `customer_segmentation.csv`, which includes the following key attributes:
- `Age`: The age of the customer.
- `Gender`: The gender of the customer.
- `Annual Income (k$)`: The annual income of the customer in thousand dollars.
- `Spending Score (1-100)`: A score assigned to customers based on their spending behavior.
- `Purchase Frequency`: The frequency of purchases made by the customer.
- `Customer Category`: Categorized labels for customers based on initial analysis.
- `Location`: The geographical location of the customer.

## Project Steps
1. **Data Loading & Exploration**
   - Read the dataset and check for missing values.
   - Generate summary statistics.
2. **Exploratory Data Analysis (EDA)**
   - Visualizations for age, income, and spending score distributions.
   - Scatter plot for spending score vs. income.
3. **Feature Scaling**
   - Standardize `Annual Income`, `Spending Score`, and `Purchase Frequency` for clustering.
4. **K-Means Clustering**
   - Apply the elbow method and silhouette score to determine the optimal number of clusters.
   - Train K-Means clustering model with the selected number of clusters.
   - Assign customers to clusters and visualize results.
5. **Demographic & Geographical Insights**
   - Analyze gender distribution across customer categories.
   - Evaluate spending score variations based on location.
6. **Save Processed Data**
   - Export the segmented dataset as `processed_customer_segmentation.csv`.

## Requirements
Ensure you have the following dependencies installed before running the script:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Script
To execute the script, run:
```bash
python new.py
```
This will generate visualizations, display key insights, and save the processed dataset.

## Results
- The project identifies distinct customer groups based on spending patterns.
- Visualizations help understand the behavior of different segments.
- The output `processed_customer_segmentation.csv` contains the clustered data for further analysis.


