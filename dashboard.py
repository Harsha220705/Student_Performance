import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ================== Load Dataset ==================
st.title("Student Performance Analysis Dashboard 📊")
temp_df = pd.read_csv('student_habits_performance.csv')
# st.subheader("📂 Dataset Preview")
# st.dataframe(temp_df.head())


# ================== Data Preprocessing ==================
def preprocess_data(df):

    # Fill missing values in parental education level with mode
    df['parental_education_level'] = df['parental_education_level'].fillna(
        df['parental_education_level'].mode()[0]
    )

    # Creating new Social Media feature
    df['Social Media'] = (df['social_media_hours'] + df['netflix_hours']) // 2

    # Drop unwanted columns
    df = df.drop(columns=[
        'age', 'attendance_percentage', 'diet_quality', 'sleep_hours',
        'internet_quality', 'student_id', 'parental_education_level',
        'extracurricular_participation', 'netflix_hours', 'social_media_hours'
    ])

    return df


# ================== Model Transform + Train Split ==================
def perform_train_test(df):
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    categorical_cols = ['gender', 'part_time_job']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


# ================== Main App ==================
def main():
    df = preprocess_data(temp_df)

    X_train, X_test, y_train, y_test, preprocessor = perform_train_test(df)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)


    # st.subheader("📌 Order of Transformed Features")
    # transformed_feature_names = preprocessor.get_feature_names_out()
    # for idx, col in enumerate(transformed_feature_names):
    #     st.write(f"{idx+1}. {col}")

    st.subheader("🧍 Enter Student Details")

    # ---------- User Inputs ----------
    student_id = st.text_input("Student ID: ")
    gender = st.radio("Gender:", ('Male', 'Female', 'Other'),horizontal=True)
    part_time = st.radio("Part Time Job:", ('Yes', 'No'), horizontal=True)
    study_hours = st.number_input("Study Hours Per Day:", min_value=0, max_value=23)
    social_media = st.slider("Daily Social Media Usage (hrs):", 0.0, 10.0)
    exercise_frequency = st.number_input("Workout Days Per Week:", 0, 7)
    mental_health_rating = st.slider("Mental Health Rating:", 0.0, 10.0)
    if gender == 'Male':
        gender,other  = 1.0,0.0
    elif gender == 'Other':
        gender,other = 0.0,1.0
    else:
        gender,other = 0.0,0.0

    # (Prediction button will be added next)
    st.info("Predicted Score (Dont get offend it's just small project LOL!!)")
    input_df = pd.DataFrame(
        {
            'gender':[gender],
            'other':[other],
            'part_time':[1.0 if part_time == 'Yes' else 0.0],
            'study_hours':[study_hours],
            'social_media':[social_media],
            'exercise_frequency':[exercise_frequency],
            'mental_health_rating':[mental_health_rating]
        }
    )
    clicked = st.button("Predict Score: ")
    if clicked:

        predicted_score = model.predict(input_df)[0]
        if predicted_score > 100:
            predicted_score = 100
        st.success(f"📌 Predicted Exam Score: {predicted_score:.2f}")


    # st.dataframe(input_df)
if __name__ == '__main__':
    main()