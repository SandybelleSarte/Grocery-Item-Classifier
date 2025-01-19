import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and vectorizer
model = joblib.load('grocery_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define categories
categories = ['Produce', 'Meat and Seafood', 'Dairy', 'Pantry Staples', 'Bakery', 'Snacks', 'Beverages', 'Frozen Foods', 'Health and Beauty', 'Baby and Kids', 'Household', 'Pet Supplies' ]

# Create a dictionary to map categories to images
category_images = {
    'Produce': 'category_images/produce.jpg',
    'Meat and Seafood': 'category_images/meat_and_seafood.jpg',
    'Dairy': 'category_images/dairy.jpg',
    'Pantry Staples': 'category_images/pantry_staples.jpg',
    'Bakery': 'category_images/bakery.png',
    'Beverages': 'category_images/beverages.jpg',
    'Snacks': 'category_images/snacks.jpg',
    'Frozen Foods': 'category_images/frozen_foods.jpg',
    'Health and Beauty': 'category_images/health_and_beauty.jpg',
    'Baby and Kids': 'category_images/baby_and_kids.jpg',
    'Household': 'category_images/household.jpg',
    'Pet Supplies': 'category_images/pet_supplies.jpeg'
}

# Load the dataset to check user input
data = pd.read_csv('grocery_dataset.csv')
grocery_items = data['Grocery Item'].tolist()

# Load the test dataset for evaluation
data = pd.read_csv('grocery_dataset.csv')
X_test = data['Grocery Item']
y_test = data['Category']

# Preprocess the test data
X_test_vec = vectorizer.transform(X_test)

# Streamlit app
st.title('Grocery Item Classifier')

# User input
grocery_item = st.text_input('Enter a grocery item:')

if grocery_item:
    # Check if the input is in the dataset
    if grocery_item.lower() in [item.lower() for item in grocery_items]:
        # Preprocess user input
        input_vec = vectorizer.transform([grocery_item])
        
        # Make prediction
        prediction = model.predict(input_vec)
        category = prediction[0]
        
        # Display the result
        st.write(f'Category: {category}')
        
        # Display the corresponding image
        image_path = category_images[category]
        image = Image.open(image_path)
        st.image(image)
    else:
        # Display a message if the input is not in the dataset
        st.write("Sorry, that isn't in our system")
    
# Display performance metrics
if st.button('Show Model Performance'):
    # Make predictions on the test data
    y_pred = model.predict(X_test_vec)
    
    # Generate performance metrics
    report = classification_report(y_test, y_pred, target_names=categories)
    st.text('Classification Report:')
    st.text(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)
    
    # Display overall accuracy
    accuracy = np.mean(y_pred == y_test)
    st.write(f'Overall Accuracy: {accuracy * 100:.2f}%')
    
    # Visualizing category distribution 
    st.subheader('Category Distribution of Test Data')
    category_counts = y_test.value_counts()
    st.bar_chart(category_counts)