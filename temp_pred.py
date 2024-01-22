# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:39:06 2023

@author: A
"""

import pickle
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


load=open('motorspeedpredication.pkl','rb')
model=pickle.load(load)
st.set_page_config(page_title="Electric Motor Speed Predicition", page_icon='https://img.icons8.com/?size=512&id=GV2UONtE91LU&format=png')
st.markdown("""
    <style>
        /* Set background image */
        .stApp {
            background-image: url("https://accelontech.com/wp-content/uploads/2022/08/industries-bg-electronics.jpg");
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* Center the title */
        .title {
            text-align: center;
            border-radius: 50px;
            color: rgb(255 255 255);
            background: #0894ca8f;

        }
        
        /* Increase the size of text inputs */
        .stNumberInput label p{
            padding: 10px;
            font-size: 20px;
       }

       /* Increase the size of number inputs */
       .stNumberInput input[type="number"] {
           padding: 10px;
           font-size: 18px;
           }
       
       /* Prediction output  */ 
       .css-nahz7x p{
            font-size: 18px   
        }
       
       .st-b7 {
           color: black;
           font-size: 22px
       }
       
       .st-ch {
           background-color: rgb(255 255 255 / 70%) !important;
       }
       
       /* Upload CSV design */
       
       .stMarkdown{
           margin-top: 35px;
           margin-bottom: 25px;

           }
            
       #upload-csv-file{
           text-align: center;
           color: #dcdcdc;
           font-size: 32px;
           background: #31c0238f;
           border-radius: 50px;
           padding: 8px;
           margin-bottom: 25px;
       }
            
        .stAlert{
            background: rgb(255 255 255 / 65%);
            border-radius: 15px;
            }
            
     /* Input label color */
     .css-1qg05tj{
         color: rgb(164 209 234);
         }
    </style>
    <h1 class="title">Electric Motor Speed Prediction</h1>
""", unsafe_allow_html=True)


def predict (inputs):
    prediction=model.predict([inputs])
    return prediction


    
def get_file_name():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name = 'Predicted_Data_'+ formatted_datetime + '.csv'
    return file_name



def main():
    page = st.sidebar.radio("Navigate", ("Description","Electric Motor Prediction", "Visualization"))
    
    if page == "Description":
        display_description_page()
    elif page == "Electric Motor Prediction":
        display_electric_motor_prediction_page()
    elif page == "Visualization":
        display_visualization_page()
        

def display_electric_motor_prediction_page():
    coolant = st.number_input("Motor Coolant",  format="%.4f")
    u_d = st.number_input("D component of Voltage",  format="%.4f")
    u_q = st.number_input("Q component of Voltage", format="%.4f")
    i_d = st.number_input("D component of Current",format="%.4f")
    i_q = st.number_input("Q component of Current", format="%.4f")

    if st.button('Predict'):
        result=predict([coolant,u_d,u_q,i_d,i_q])
        st.success('The motor speed is  {}'.format(result[0]))
    
    
    st.subheader('Upload CSV file')
    
    #csv input
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV data using 
        index_names_to_read = ['coolant','u_d','u_q','i_d','i_q']
        df = pd.read_csv(uploaded_file, usecols= lambda column: column in index_names_to_read)
        # Display the data frame
        
        predicted_values = []
        for index, row in df.iterrows():
            row_array = row.to_numpy()
            predicted_value = predict(row_array);
            predicted_values.append(predicted_value[0])
            

       # Display the updated DataFrame
        df['Predicted Motor Speed'] = predicted_values
        csv_file = df.to_csv(index=False)
        file_name = get_file_name()
        st.session_state['predicted_data'] = df;
        st.dataframe(df)
        st.download_button("Download CSV", data=csv_file, file_name = file_name, mime='text/csv')
        
def display_visualization_page():
    option = st.sidebar.radio("Select an option for Visualization", ("Histogram", "Boxplot", "Correlation plot", "Scatter plot"))

    # Display content based on the selected option
    if option == "Histogram":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_histogram()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
    elif option == "Boxplot":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_boxplot()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
        
    elif option == "Correlation plot":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_correlation_chart()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
        
    elif option == "Scatter plot":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_scatter_chart()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
        
        
def display_description_page():
    st.subheader('Description')
    st.write("In this electric motor speed prediction model, I used LightGBM as your machine learning algorithm. The goal was to predict motor speed, and to achieve this, I selected specific features that were deemed important for the prediction. These features include the coolant, d component of current, and q component of voltage.")
    st.write("The coolant is likely a crucial factor in determining the motor's efficiency and performance, while the d and q components of current and voltage are essential components in understanding the motor's electromagnetic behavior.")
    st.write("By training my LightGBM model on this selected set of features, I aimed to capture the complex relationships between these variables and the motor speed. The LightGBM algorithm is known for its efficiency and accuracy in handling large datasets, making it a suitable choice for this task.")
    st.write("The outcome of my prediction model is expected to provide valuable insights into the motor's speed based on the given input features, enabling better control and optimization of electric motor systems.")
    
    
    
def show_histogram():
    #global pred_data
    st.title("Histogram")
    predicted_data = st.session_state['predicted_data']
    
    select_feature = st.sidebar.selectbox("Select Feature", ['coolant','u_d','u_q','i_d','i_q'])
    if select_feature == 'coolant':
        data = predicted_data['coolant'].values
    elif select_feature == 'u_d':
        data = predicted_data['u_d'].values
    elif select_feature == 'u_q':
        data = predicted_data['u_q'].values
    elif select_feature == 'i_d':
        data = predicted_data['i_d'].values
    elif select_feature == 'i_q':
        data = predicted_data['i_q'].values
    
    plt.hist(data, bins=20, density=False, color='blue', alpha=0.7)

   # Add labels and title to the plot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram for feature: {select_feature}")
    st.pyplot()

def show_boxplot():
    st.title("Boxplot")
    predicted_data = st.session_state['predicted_data']
    print(predicted_data.values)
    select_feature = st.sidebar.selectbox("Select Feature", ['coolant','u_d','u_q','i_d','i_q'])
    if select_feature == 'coolant':
        data = predicted_data['coolant'].values
    elif select_feature == 'u_d':
        data = predicted_data['u_d'].values
    elif select_feature == 'u_q':
        data = predicted_data['u_q'].values
    elif select_feature == 'i_d':
        data = predicted_data['i_d'].values
    elif select_feature == 'i_q':
        data = predicted_data['i_q'].values
    
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title(f'Boxplot for {select_feature}')

    # Show the boxplot using Streamlit
    st.pyplot(fig)
    
def show_correlation_chart():
    st.title("Correlation Heatmap")
    predicted_data = st.session_state['predicted_data']
    # Calculate the correlation matrix
    corr_matrix = predicted_data.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot()
    
def show_scatter_chart():
    st.title("Scatter Chart")
    predicted_data = st.session_state['predicted_data']
    
    select_feature = st.sidebar.selectbox("Select Feature", ['coolant','u_d','u_q','i_d','i_q'])
    if select_feature == 'coolant':
        data = predicted_data['coolant'].values
    elif select_feature == 'u_d':
        data = predicted_data['u_d'].values
    elif select_feature == 'u_q':
        data = predicted_data['u_q'].values
    elif select_feature == 'i_d':
        data = predicted_data['i_d'].values
    elif select_feature == 'i_q':
        data = predicted_data['i_q'].values
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data, predicted_data['Predicted Motor Speed'], c='blue', marker='o')
    plt.xlabel('Motor Speed')
    plt.ylabel(f'{select_feature}')
    plt.title(f'Scatter Plot for {select_feature}')
    st.pyplot()
    
        
if __name__=='__main__':
    main()