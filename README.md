# 🛒 Flipkart GRiD 6.0 

## 1. Freshness Detection of Fruits & Vegetables

### 📌 Project Overview
This project was built as part of **Flipkart Grid 4.0**, focusing on detecting the freshness of **fruits, vegetables, and leaves** using **computer vision and image processing techniques**. The goal was to develop a system that could assess freshness levels based on visual cues like **color, texture, and spoilage patterns**.

### 🚀 Features
- **Freshness Detection:** Classifies produce into five categories: **100% fresh, 75% fresh, 50% fresh, 25% fresh, and rotten**.
- **Computer Vision-Based Analysis:** Uses **OpenCV** for real-time feature extraction and classification.
- **Custom Dataset:** Built a dataset of real images since publicly available datasets were not suitable for real-world application.
- **Dynamic Spoilage Criteria:** Applied different freshness assessment rules based on **fruit and vegetable-specific spoilage patterns**.
- **Scalability:** Designed the system to allow further integration with **real-time market APIs** for better accuracy.

### 🏗 Tech Stack
- **Languages:** Python
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib
- **Computer Vision**
- **Frameworks:** Streamlit (for UI visualization)

### 🔍 How It Works
1. **Image Preprocessing:** Converts images to grayscale and applies edge detection to highlight spoilage features.
2. **Feature Extraction:** Identifies spoilage based on visual cues like **color fading, texture changes, and mold formation**.
3. **Freshness Classification:** Uses predefined thresholds for different produce types to determine freshness.
4. **User Interface:** Displays classification results via a simple **Streamlit web app**.

### 📊 Results & Performance
- Successfully classified **freshness levels** with **high accuracy** on real-world test data.

### 💡 Key Takeaways
- **Custom feature engineering** was key to improving real-world classification accuracy.
- **Rule-based classification worked better than deep learning models** due to limited real-world training data.
- **AI in grocery supply chains** can help reduce food waste and improve inventory management.

> **Checkout the demo video here:** [Freshness Detection Demo](path_to_freshness_detection_demo.MP4)


<p align="left">
  <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTJoa2d3eTltbnFxdzg3ODdoazI2bHRyOWszODhob3AycGR4NXF0ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xOufNMIlQeiEmJhtqO/giphy.gif" width="300"/>
</p>
