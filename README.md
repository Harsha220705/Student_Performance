# Student Performance Prediction Dashboard 📊

An interactive web-app that predicts a student’s exam score based on their lifestyle and habits using machine learning. Built with Python, Streamlit, and Scikit-learn.

## 🔍 Overview

This application allows users to input daily behaviors and personal habits such as:

- Study hours per day  
- Social media usage  
- Exercise frequency  
- Mental health rating  
- Gender  
- Part-time job status

Based on these inputs, the app uses a trained Linear Regression model to estimate the student’s exam score in real time.

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn (data preprocessing & model)  
- Streamlit (interactive UI & deployment)  

## ⚙️ Setup & Run

```bash
# Clone the repo
git clone https://github.com/Harsha220705/Student_Performance.git
cd Student_Performance

# (Optional) Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard.py
