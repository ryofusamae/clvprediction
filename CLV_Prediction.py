import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import streamlit as st
import logging
import os
import matplotlib.pyplot as plt

# Set the basic configuration for the logger
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Check if dataset exists before trying to load it
dataset_path = 'marketing_campaign.csv'
if not os.path.exists(dataset_path):
    logging.error("Dataset not found.")
    raise FileNotFoundError("Dataset not found.")

# Load Database and log any errors
try:
    df = pd.read_csv(dataset_path, sep='\t')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the dataset. Error: {str(e)}")
    raise

# Get the list of all column names from headers
column_headers = df.columns.to_list()
print("The Column Header :", column_headers)

# Create a 'TotalSpent' column as our target variable
df['TotalSpent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

# Convert 'Dt_Customer' to a numeric feature: number of days since enrollment
df['Dt_Customer'] = (pd.to_datetime('today') - pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y")).dt.days

# We'll drop the columns that we've transformed or that aren't needed for our prediction
df = df.drop(columns=['ID', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])

# One-hot encode categorical variables
df = pd.get_dummies(df)

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Then apply imputation to the entire dataset and convert back to dataframe
# df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

# Split the dataset into training and test sets
X = df.drop(columns=["TotalSpent"])
y = df["TotalSpent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f'Training set size: {len(X_train)}, Test set size: {len(X_test)}.')

# Train the imputer on the training data only
imputer.fit(X_train)

# Apply imputation on X_train and X_test datasets
X_train = pd.DataFrame(imputer.transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
logging.info('Random Forest model trained successfully.')

# Save the trained model
joblib.dump(model, 'clv_model.pkl')
logging.info('Trained model saved to clv_model.pkl.')

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
logging.info(f'Mean Absolute Error: {mae:.2f}')
logging.info(f'Root Mean Squared Error: {rmse:.2f}')

# Fetch the maximum and minimum of the TotalSpent in the entire dataset
max_spent = df['TotalSpent'].max()
min_spent = df['TotalSpent'].min()

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Max Value of TotalSpent: {max_spent:.2f}")
print(f"Min Value of TotalSpent: {min_spent:.2f}")


# Predict total spending over next 2 years
predicted_spending_2yrs = model.predict(X_test)

# Predict CLV
average_lifespan = 5  # This is an example value, you'd use what makes sense for your business
predicted_clv = (average_lifespan / 2) * predicted_spending_2yrs

#Descriptive Statistics: Get basic stats about the predicted CLV.
logging.info(f"Average Predicted CLV: {predicted_clv.mean()}")
logging.info(f"Median Predicted CLV: {np.median(predicted_clv)}")
logging.info(f"Max Predicted CLV: {predicted_clv.max()}")
logging.info(f"Min Predicted CLV: {predicted_clv.min()}")

# CLV Segmentation: Classify customers into segments based on their predicted CLV.
def clv_segmentation(value):
    if value <= 500:
        return "# of Low Value Customer"
    elif value <= 1000:
        return "# of Medium Value Customer"
    else:
        return "# of High Value Customer"

X_test['Predicted_CLV'] = predicted_clv
X_test['CLV_Segment'] = X_test['Predicted_CLV'].apply(clv_segmentation)
print(X_test['CLV_Segment'].value_counts())

# Find Top X% of Customers: Identify the top customers based on their predicted CLV.
top_10_percent = np.percentile(predicted_clv, 90)
top_customers = X_test[X_test['Predicted_CLV'] > top_10_percent]
print("Top 10% customers based on Predicted CLV:", top_customers.shape[0])

# Relationship between CLV and Income shown as scattered plot
sns.scatterplot(x=X_test['Income'], y=X_test['Predicted_CLV'])
plt.title('Relationship between Income and Predicted CLV')
plt.show()

# Kidhome Analysis:

# Distribution of Households by Number of Kids
sns.countplot(x='Kidhome', data=df)
plt.title("Distribution of Households by Number of Kids")
plt.show()

# Average Spend by Kidhome Category
sns.barplot(x='Kidhome', y='TotalSpent', data=df, estimator=sum)
plt.title("Total Spend by Number of Kids in Household")
plt.show()

# 2. Purchase Patterns:
# Load the dataset
df_original = pd.read_csv('marketing_campaign.csv', sep='\t')

# Create a 'TotalSpent' column as our target variable
df_original['TotalSpent'] = df_original['MntWines'] + df_original['MntFruits'] + df_original['MntMeatProducts'] + df_original['MntFishProducts'] + df_original['MntSweetProducts'] + df_original['MntGoldProds']

# a. Most Popular Products
categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
total_spends = [df_original[cat].sum() for cat in categories]
plt.bar(categories, total_spends)
plt.title("Total Spend per Product Category")
plt.xticks(rotation=45)
plt.show()

# b. Purchase Frequency using Recency as a proxy
# Assuming a customer purchases once every time they visit, hence the frequency is approximately the inverse of recency.
df_original['Recency'] = df_original['Recency'].replace(0, 1)  # replace 0 with 1 to prevent division by zero
df_original['Purchase_Frequency'] = 1 / df_original['Recency']
sns.histplot(df_original['Purchase_Frequency'], kde=False, bins=30)
plt.title("Distribution of Purchase Frequency (using Recency as a proxy)")
plt.show()

# Customer Demographics:
# Calculate Age
current_year = 2023  # Adjust this according to your dataset's current year
df_original['Age'] = current_year - df_original['Year_Birth']

# Plot
sns.histplot(df_original['Age'], kde=False, bins=30)

# Age Distribution
sns.histplot(df_original['Age'], kde=False, bins=30)
plt.title("Age Distribution of Customers")
plt.show()

# Income Distribution
sns.histplot(df_original['Income'], kde=False, bins=30)
plt.title("Income Distribution of Customers")
plt.show()


def login():
    # Check if 'logged_in' is already set in the session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Initialize session state

    if st.session_state.logged_in:
        st.sidebar.success("Logged in successfully.")
        return True

    st.sidebar.title("Login")

    # Hardcoded username and password (For demonstration purposes only)
    correct_username = "admin"
    correct_password = "password123"

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if username and password:
        if username == correct_username and password == correct_password:
            st.session_state.logged_in = True
            st.sidebar.success("Logged in successfully.")
            return True
        else:
            st.sidebar.warning("Incorrect Username/Password")

    return False

# Check if user is authenticated
is_authenticated = login()

if is_authenticated:

    # STREAMLIT INTERACTIVE CLV PREDICTOR
    # Load the saved model
    model = joblib.load('clv_model.pkl')

    st.title("Interactive CLV Predictor")

    # Sidebar for data exploration
    st.sidebar.header('Data Exploration')
    if st.sidebar.checkbox('Show raw data'):
        st.subheader('Marketing Campaign Data')
        st.write(df)

    # User Input Collection
    st.subheader("Enter Customer Details")

    with st.expander("Purchase Information"):
        recency = st.slider('Number of days since last purchase:', 0, 365, 50)
        num_web_purchases = st.slider('Number of purchases made through the company’s website:', 0, 50, 10)

    with st.expander("Personal Details"):
        income = st.number_input("Enter Income")
        # Sanitize user input: Ensure that it's a number, avoid negatives, etc.
        if income < 0:
            st.warning("Income should be a positive number.")
            st.stop()

    user_data_dict = {
        "Recency": [recency],
        "NumWebPurchases": [num_web_purchases],
        "Income": [income]
        # Add other features here as well
    }
    columns_order = X_train.columns.tolist()
    user_data = pd.DataFrame(user_data_dict, columns=columns_order)

    # Preprocess user data
    user_data_imputed = pd.DataFrame(imputer.transform(user_data), columns=user_data.columns)

    # Predict CLV
    predicted_clv = model.predict(user_data_imputed)
    st.subheader(f"Predicted Customer Lifetime Value (CLV): ${predicted_clv[0]:.2f}")

    # Visualization of CLV distribution
    st.subheader("CLV Distribution in Dataset")
    fig, ax = plt.subplots()
    sns.histplot(df["TotalSpent"], kde=False, bins=30, ax=ax)
    ax.axvline(predicted_clv[0], color='red', linestyle='dashed', linewidth=2)
    ax.set_title("Where Your CLV Stands")
    st.pyplot(fig)

    # Visualization of CLV Segments
    segments = df["TotalSpent"].apply(clv_segmentation).value_counts()
    st.subheader("CLV Segmentation in Dataset")
    fig, ax = plt.subplots()
    segments.plot(kind='bar', ax=ax)
    ax.set_title("Distribution of Customers by CLV Segment")
    st.pyplot(fig)

    st.subheader("Other Relevant Data")
    # Scatter Plot - Relationship between CLV and Income
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_test['Income'], y=X_test['Predicted_CLV'])
    plt.title('Relationship between Income and Predicted CLV')
    st.pyplot(fig)
    # Histogram - Distribution of Households by Number of Kids
    fig, ax = plt.subplots()
    sns.countplot(x='Kidhome', data=df)
    plt.title("Distribution of Households by Number of Kids")
    st.pyplot(fig)
    # Bar Plot - Total Spend by Number of Kids in Household
    fig, ax = plt.subplots()
    sns.barplot(x='Kidhome', y='TotalSpent', data=df, estimator=sum)
    plt.title("Total Spend by Number of Kids in Household")
    st.pyplot(fig)
    # Bar Plot - Total Spend per Product Category
    fig, ax = plt.subplots()
    categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    total_spends = [df_original[cat].sum() for cat in categories]
    plt.bar(categories, total_spends)
    plt.title("Total Spend per Product Category")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # Histogram - Distribution of Purchase Frequency (using Recency as a proxy)
    fig, ax = plt.subplots()
    sns.histplot(df_original['Purchase_Frequency'], kde=False, bins=30)
    plt.title("Distribution of Purchase Frequency (using Recency as a proxy)")
    st.pyplot(fig)
    # Histogram - Age Distribution of Customers
    fig, ax = plt.subplots()
    sns.histplot(df_original['Age'], kde=False, bins=30)
    plt.title("Age Distribution of Customers")
    st.pyplot(fig)

else:
    st.warning("Please log in to access the app.")