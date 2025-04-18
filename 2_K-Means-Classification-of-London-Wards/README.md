
# 🗂️ K-Means Classification of London Wards

## 📌 Objective
This project aims to group London wards based on their land use percentages using the **K-Means clustering algorithm**. The resulting clusters represent different urban typologies.

## 🔧 Methods Used
- **EDA** (Exploratory Data Analysis)
- **Feature Scaling** using StandardScaler
- **K-Means Clustering** (with Elbow Method + Evaluation Metrics)
- **Spatial Data Integration**: Merging ward-level CSV data with shapefiles containing London boundary data
- **Map Generation** using GeoPandas and Matplotlib to visualize the spatial distribution of clusters
- **Visualization and Export of Results**

## 📊 Dataset
Each ward is described by the percentage of the following land use types:
- Domestic / Non-Domestic Buildings  
- Roads, Railways, Paths  
- Greenspace, Water Bodies, Other Land Uses

## 📎 Outputs
- `2_London_wards_with_labels`: Cluster labels assigned to each ward  
- `2_K-Means-Classification-of-London-Wards.ipynb`: Complete analysis workflow

## 👤 Author
**Naile Yalgettekin**  
[GitHub](https://github.com/yalgettekin)

