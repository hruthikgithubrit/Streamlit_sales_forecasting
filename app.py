import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import datetime

le_store = LabelEncoder()
le_item = LabelEncoder()

# Load your sales data (replace 'your_data.csv' with your actual data file)
@st.cache_data(ttl=3600)
def load_data():
    data = pd.read_csv('train.csv')
    return data

# Preprocess the data
def preprocess_data(data):
    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Label encode categorical columns
    #le = LabelEncoder()
    data['store'] = le_store.fit_transform(data['store'])
    data['item'] = le_item.fit_transform(data['item'])

    return data

# Train a simple model (you should replace this with a more sophisticated model)
def train_model(data):
    X = data[['store', 'item']]
    y = data['sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor (you should replace this with your model)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    return model

# Sales prediction function
def predict_sales(model, le_store, le_item, selected_store, selected_item):
    try:

      selected_store_encoded = le_store.transform([selected_store])[0]
      selected_item_encoded = le_item.transform([selected_item])[0]
    except ValueError as e:
      #handle the case where a label is unseen
      st.error(f"Error: {e}")
      return None
    
    prediction_data = pd.DataFrame({'store': [selected_store_encoded], 'item': [selected_item_encoded]})

    prediction = model.predict(prediction_data)
    # Inverse transform to get original labels
    predicted_store = le_store.inverse_transform([selected_store_encoded])[0]
    predicted_item = le_item.inverse_transform([selected_item_encoded])[0]

    return prediction[0], predicted_store, predicted_item

# Streamlit app
def main():
    st.title('Sales Forecasting App')

    # Load data
    data = load_data()

    # Preprocess data
    data = preprocess_data(data)

    # Train model
    model = train_model(data)

    # Sidebar with user input
    st.sidebar.header('User Input')
    #selected_date = st.sidebar.date_input('Select Date', min_value=data['date'].min(), max_value=data['date'].max())
    # Assuming min_date and max_date are calculated based on your data
    min_date = data['date'].min()
    max_date = data['date'].max()

    # Setting a default value within the specified range
    default_date = min_date + (max_date - min_date) // 2
    selected_date = st.sidebar.date_input('Select Date', min_value=min_date, max_value=max_date, value=default_date)

    selected_store = st.sidebar.selectbox('Select Store', data['store'].unique())
    selected_item = st.sidebar.selectbox('Select Item', data['item'].unique())

    # Filter data based on user input
    #filtered_data = data[(data['date'] == selected_date) & (data['store'] == selected_store) & (data['item'] == selected_item)]

    # Display filtered data
    #st.subheader('Filtered Data')
    #st.write(filtered_data)

    # Make sales prediction
    #if not filtered_data.empty:
        #predicted_sales = predict_sales(model, selected_store, selected_item)
        #st.subheader('Sales Prediction')
        #st.write(f'Predicted Sales: {predicted_sales}')

    # ... (previous code)

    # Filter data based on user input
    filtered_data = data[(data['date'] == pd.to_datetime(selected_date)) & (data['store'] == selected_store) & (data['item'] == selected_item)]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Convert selected store and item using label encoding
    selected_store_encoded = le_store.transform([selected_store])[0]
    selected_item_encoded = le_item.transform([selected_item])[0]

    # Make sales prediction
    if not filtered_data.empty:
        predicted_sales, predicted_store, predicted_item = predict_sales(model, le_store, le_item, selected_store_encoded, selected_item_encoded)
        if predicted_sales is not None:
          st.subheader('Sales Prediction')
          st.write(f'Predicted Sales: {predicted_sales}')
        else:
          st.warning("Unable to make prediction due to unseen label.")


if __name__ == '__main__':
    main()

