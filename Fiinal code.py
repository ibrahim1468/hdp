import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Streamlit App
st.title("Heart Disease Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    try:
        # Read and validate the dataset
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        if "Severity" not in data.columns:
            st.error("The dataset must contain a 'Severity' column.")
        else:
            # Handle missing values
            if data.isnull().sum().sum() > 0:
                st.warning("The dataset contains missing values. Rows with missing values will be removed.")
                data = data.dropna()

            # Prepare data
            X = data.drop(columns=["Severity"], axis=1)
            y = data["Severity"]

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model selection
            model_choice = st.selectbox("Select a model", ["KNeighborsClassifier", "RandomForestClassifier", "SVC"])

            models = {
                "KNeighborsClassifier": {
                    "model": KNeighborsClassifier(),
                    "parameters": {"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"], "p": [1, 2]},
                    "scale": False,
                },
                "RandomForestClassifier": {
                    "model": RandomForestClassifier(),
                    "parameters": {"n_estimators": [50, 100, 200], "criterion": ["entropy", "gini"]},
                    "scale": False,
                },
                "SVC": {
                    "model": SVC(class_weight="balanced"),
                    "parameters": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": [0.01, 0.1, 1]},
                    "scale": True,
                },
            }

            selected_model = models[model_choice]
            X_train_to_use = X_train_scaled if selected_model["scale"] else X_train
            X_test_to_use = X_test_scaled if selected_model["scale"] else X_test

            # Train model with hyperparameter tuning
            st.write("Tuning hyperparameters...")
            try:
                search = GridSearchCV(estimator=selected_model["model"], param_grid=selected_model["parameters"], cv=5, n_jobs=-1)
                search.fit(X_train_to_use, y_train)

                # Evaluation
                predictions = search.predict(X_test_to_use)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
                f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

                st.write(f"Best Parameters: {search.best_params_}")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Input form for single instance
                st.write("### Predict on a single instance")
                input_data = {col: st.number_input(col, value=0.0) for col in X.columns}
                input_df = pd.DataFrame([input_data])
                input_df_aligned = input_df[X.columns]
                input_scaled = scaler.transform(input_df_aligned) if selected_model["scale"] else input_df_aligned

                prediction = search.predict(input_scaled)
                severity_map = {
                    0: "You may have no heart disease",
                    1: "You may have mild heart disease",
                    2: "You may have moderate heart disease",
                    3: "You may have severe heart disease",
                    4: "You may be in advanced stage",
                }
                st.write(f"Prediction for the input instance: {prediction[0]}")
                st.write(severity_map.get(prediction[0], "Unknown severity level"))

            except Exception as e:
                st.error(f"An error occurred during model training: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
