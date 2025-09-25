import streamlit as st
import joblib
import pandas as pd

# Load the saved model and scaler
# Ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory or provide the full path
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model or scaler. Make sure 'best_model.pkl' and 'scaler.pkl' are in the correct directory.")
    st.stop() # Stop execution if files are not found

# Assuming you have a list of columns used during training (after one-hot encoding)
# It's crucial that the columns in the input data match the columns the model was trained on.
# You should save X_train.columns during training and load it here.
# For demonstration, let's assume a simplified list of columns.
# Replace this with the actual columns from your X_train DataFrame.
# You can get these columns from the variable `X` after one-hot encoding in your training notebook.
# Example: training_columns = X.columns.tolist() # Save this list
# Load the training columns:
try:
    # Attempt to load saved training columns
    training_columns = joblib.load('training_columns.pkl')
    st.success("Training columns loaded successfully!")
except FileNotFoundError:
    st.warning("Could not find 'training_columns.pkl'. Using a placeholder list. Predictions may be inaccurate.")
    # In a real application, you MUST load the actual training columns.
    # This is a placeholder for demonstration purposes.
    training_columns = ['year', 'kilometerdriven', 'ownernumber', 'isc24assured', 'benefits', 'discountprice', 'created_year', 'created_month', 'created_dayofweek', 'car_age', 'month_num'] # Add your actual columns here


st.title('Car Price Prediction App')

st.write("""
This app predicts the price of a car based on its features.
Please enter the car's specifications below:
""")

# Input fields for features
# You'll need to add input fields for all features used in your model (including dummy variables)
# For categorical features, you can use selectbox or radio buttons and then one-hot encode the input.
# For simplicity, let's start with a few numerical features.
# You'll need to expand this to include all your features.

year = st.number_input('Year', min_value=1990, max_value=2023, value=2022)
kilometerdriven = st.number_input('Kilometer Driven', min_value=0, value=50000)
ownernumber = st.selectbox('Owner Number', [1, 2, 3])
isc24assured = st.checkbox('Is C24 Assured?')
benefits = st.number_input('Benefits', min_value=0, value=10000)
discountprice = st.number_input('Discount Price', min_value=0, value=5000)

# Date related features (you'll need to handle these based on user input or logic)
# For simplicity, let's use current date or derive from year.
# In a real app, you might ask for the listing date.
created_year = 2023 # Example
created_month = 1 # Example
created_dayofweek = 0 # Example (Monday)
car_age = created_year - year # Example
month_num = created_month # Example


# Create a DataFrame from the input
input_data = pd.DataFrame([[
    year, kilometerdriven, ownernumber, isc24assured, benefits, discountprice,
    created_year, created_month, created_dayofweek, car_age, month_num
    # Add values for all other features (including dummy variables) here
]], columns=['year', 'kilometerdriven', 'ownernumber', 'isc24assured', 'benefits', 'discountprice', 'created_year', 'created_month', 'created_dayofweek', 'car_age', 'month_num']) # Add your actual columns here

# --- Handling Categorical Features (Example) ---
# You need to handle all your categorical features here to create the dummy variables
# This part is crucial and depends on the categories present in your training data.
# You'll need to get the list of unique values for each categorical feature from your training data
# and create the corresponding dummy columns with 0 or 1 based on user selection.

# Example for 'fueltype' (assuming 'Petrol', 'Diesel', 'Petrol + Cng' were in training)
# fueltype = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Petrol + Cng'])
# if fueltype == 'Petrol':
#     input_data['fueltype_Petrol'] = 1
#     input_data['fueltype_Diesel'] = 0
#     input_data['fueltype_Petrol + Cng'] = 0
# elif fueltype == 'Diesel':
#     input_data['fueltype_Petrol'] = 0
#     input_data['fueltype_Diesel'] = 1
#     input_data['fueltype_Petrol + Cng'] = 0
# else: # Petrol + Cng
#     input_data['fueltype_Petrol'] = 0
#     input_data['fueltype_Diesel'] = 0
#     input_data['fueltype_Petrol + Cng'] = 1

# Repeat this for all your categorical features (make, model, city, transmission, bodytype, etc.)
# Ensure that the column names created here exactly match the column names in your X_train after one-hot encoding.

# --- Aligning columns with training data ---
# This is the most critical step. The input DataFrame must have the exact same columns
# in the exact same order as the DataFrame used to train the model (X_train).
# If any column is missing in the input (e.g., a category not present in the input),
# it must be added with a value of 0.
# The best way to handle this is to save the list of columns from your X_train DataFrame
# after one-hot encoding and use that list here.

# Example of aligning columns (assuming you loaded 'training_columns.pkl')
try:
    input_data = input_data.reindex(columns=training_columns, fill_value=0)
except NameError:
    st.error("Training columns not loaded. Cannot align input data. Please ensure 'training_columns.pkl' exists.")
    st.stop()


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button('Predict Price'):
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted Car Price: ₹{prediction[0]:,.2f}')
