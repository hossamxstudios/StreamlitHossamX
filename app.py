import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import os 
from pycaret.classification import *  
import pycaret.classification as pc 
import pycaret.regression as pr
from pycaret.classification import evaluate_model as evaluate_model
from pycaret.regression import evaluate_model as evaluate_model1
from joblib import dump, load

@st.cache_data 
def drop_missing_values():
    cleaned_data = df.copy().dropna(inplace=True)
    st.write("DataFrame after dropping missing values:")
    st.write(cleaned_data)
    
if 'drop_missing_values_btn' not in st.session_state:
    st.session_state['drop_missing_values_btn'] = False
def drop_missing_values_fun():
    st.session_state['drop_missing_values_btn'] = True
      
if 'fill_missing_values_btn' not in st.session_state:
    st.session_state['fill_missing_values_btn'] = False
def fill_missing_values_fun():
    st.session_state['fill_missing_values_btn'] = True

if 'fill_data_button' not in st.session_state:
    st.session_state['fill_data_button'] = False
def save_btn_fill():
    st.session_state['fill_data_button'] = True
    
if 'hue_button' not in st.session_state:
    st.session_state['hue_button'] = False
def hue_fun():
    st.session_state['hue_button'] = True
    
if 'train_button' not in st.session_state:
    st.session_state['train_button'] = False
def train_fun():
    st.session_state['train_button'] = True

if 'trans_button' not in st.session_state:
    st.session_state['trans_button'] = False
def trans_fun():
    st.session_state['trans_button'] = True

if 'trans_button1' not in st.session_state:
    st.session_state['trans_button1'] = False
def trans_fun1():
    st.session_state['trans_button1'] = True
    
if 'clf_button' not in st.session_state:
    st.session_state['clf_button'] = False
def train_clf_fun():
    st.session_state['clf_button'] = True
    
if 'save_model_btn' not in st.session_state:
    st.session_state['save_model_btn'] = False
def save_model_fun():
    st.session_state['save_model_btn'] = True  

if 'pred_btn' not in st.session_state:
    st.session_state['pred_btn'] = False
def pred_func():
    st.session_state['pred_btn'] = True       
      
@st.cache_data  # Cache the data
def classify_column_type(series):
    if series.dtype in ['int64', 'float64']:
        return 'Numerical'
    else:
        return 'Categorical'
    
@st.cache_data  # Cache the data
def fill_missing_values(df, fill_methods, custom_values):
    for col, method in fill_methods.items():
        if method.startswith('Fill'):
            if method == 'Fill with Mean':
                fill_value = df[col].mean()
            elif method == 'Fill with Median':
                fill_value = df[col].median()
            elif method == 'Fill with Mode':
                fill_value = df[col].mode().iloc[0]
            elif method == 'Fill with Custom':
                fill_value = custom_values[col]
            df[col].fillna(fill_value, inplace=True)
    return df

st.set_page_config(page_title="Data Analysis")
st.title("Data Analysis")

get_file = st.form('data_analysis')
file = get_file.file_uploader('Upload a data file', type=['csv', 'xlsx', 'xls'])
if file is not None:
    file_extension = file.name.split('.')[-1]
    if file_extension.lower() in ['csv', 'xlsx', 'xls']:
        if file_extension.lower() == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        column_types = {col: classify_column_type(df[col]) for col in df.columns}
        missing_values = df.isnull().sum()
    else:
        st.error('Only CSV, XLSX, or XLS files are supported.')
get_file_button = get_file.form_submit_button("Apply")
        
col3,col4 = st.columns([1, 1])

try:
    if df is not None:
        st.write(df)    
        if missing_values.sum() > 0:
            with col3:
                st.write("your dataframe has some missing values:")
                st.write(missing_values )
            with col4:
                st.write("your column types and names :")
                st.write(column_types)
                
            with st.expander("##### Missing Values Handling"):
                col1,col2 = st.columns([1, 1])
                with col1:
                    drop_missing_values_btn = st.button("Drop Missing Values Data",on_click=drop_missing_values_fun)
                    if st.session_state['drop_missing_values_btn']:
                        df.dropna(inplace=True)
                        st.write("DataFrame after dropping missing values:")
                        st.write(df)
            
                with col2:
                    fill_missing_values_btn = st.button("Fill Missing Values",on_click=fill_missing_values_fun)
                    if st.session_state['fill_missing_values_btn']:
                        fill_methods = {}  
                        custom_values = {}  
                        with st.form('custom_value_form'):
                            for col in df.columns:
                                if missing_values[col] > 0:
                                    st.write(f"Column '{col}' has missing values.")
                                    if column_types[col] == 'Numerical':
                                        fill_method = st.radio(f"Choose fill method for '{col}':", [ 'Fill with Mean', 'Fill with Median', 'Fill with Mode'])
                                        fill_methods[col] = fill_method
                                    else:
                                        fill_method = st.radio(f"Choose fill method for '{col}':", ['Fill with Mode', 'Fill with Custom'])
                                        fill_methods[col] = fill_method
                                        custom_values[col] = st.text_input(f"Enter custom value for '{col}':")
                            fill_data_button = st.form_submit_button("Apply Changes",on_click=save_btn_fill)
                            
                            if st.session_state['fill_data_button']:
                                columns = []
                                df = fill_missing_values(df.copy(), fill_methods, custom_values)
                                st.success("DataFrame is cleaned successfully ")    
        else:
            with col3:
                st.write("No missing values found in the dataframe.")  
            with col4:
                st.write("your column types and names :")
                st.write(column_types )
      
    with st.expander("##### Data Overview"):
        columns = []
        for colm in df.columns.values:
            columns.append(colm)
        with st.form('pairplot_form'):
            hue_input = st.radio(f"Choose fill hue column:", columns)
            hue_btn = st.form_submit_button("Apply",on_click=hue_fun)
            
        if st.session_state['hue_button']:
            plot = sns.pairplot(df, hue=hue_input)
            st.pyplot(plot.figure)

    with st.expander("##### Model Training with Pycaret"):

        train_data_button = st.button("Train your Data",on_click=train_fun)
        if st.session_state['train_button']:
            columns = []
            for colm in df.columns.values:
                columns.append(colm)
                
            with st.form('train_data_form'):   
                st.session_state['hue_button'] = False
                target_column = st.radio(f"Choose target column:", columns)
                trans_btn = st.form_submit_button("Apply",on_click=trans_fun)

            if st.session_state['trans_button']:
                if df[target_column].dtype in ['int64', 'float64']:
                    task="regression"
                else:
                    task="classification"
                    
                st.write(f"Detected task type: {task}")
                if task=="classification":
                    clf=clf = pc.setup(df, target=target_column,  normalize=True, transformation=True)
                    best_model=pc.compare_models()
                    classification_result = pc.pull()
                    st.write(classification_result)
                    pc.plot_model(estimator=best_model, display_format='streamlit')
                elif  task=="regression":   
                    reg=clf = pr.setup(df, target=target_column,  normalize=True, transformation=True)
                    best_model=pr.compare_models()
                    regression_result = pr.pull()
                    st.write(regression_result)
                    pr.plot_model(estimator=best_model, display_format='streamlit')
                else:
                    st.write("Unsupported task type. Please check the target column data type.")
                
    with st.expander("##### Create and Save Models"):
        with st.form('save_model_form'):  
            save_model_btn = st.form_submit_button("Apply",on_click=save_model_fun)
        if st.session_state['save_model_btn']:
            model_folder = f"Models2/"
            model_filename = f"{best_model}.joblib"
            isExist = os.path.exists(model_folder)
            if not isExist:
                os.makedirs(model_folder)
                st.write(f"The new directory {model_folder} is created!")
            dump(best_model, f"{model_folder}{best_model}.joblib")
            st.write(f"Model {best_model} is saved successfully") 
            st.write(f"The best model is saved as {model_folder + model_filename}")
            st.write(f"Model {best_model} is saved successfully") 
            if task=="classification":
                predictions = pc.predict_model(best_model,data=df) 
                st.write(predictions.head())
            elif  task=="regression":   
                predictions = pr.predict_model(best_model,data=df) 
                st.write(predictions.head())
            else:
                st.write("Unsupported task type. Please check the target column data type.")
                        
    with st.sidebar:
        if st.session_state['save_model_btn']:
            user_inputs = {}
            st.subheader("Enter values for prediction:")
            for column in df.columns:
                if column != target_column:
                    if df[column].dtype == "object":  # Categorical column with limited choices
                        user_inputs[column] = st.selectbox(f"Select {column}", options=df[column].unique())
                    else:  # Non-categorical column or categorical column with many choices
                        user_inputs[column] = st.text_input(f"Enter value for {column}", value=df[column].iloc[0])
            pred_btn = st.button("Predict",on_click=pred_func)
        else:
            st.write('#### Create and Save your model first')
            st.write('please Create and Save your model first to be able to see the preditcion inputs')
        
    if st.session_state['pred_btn']:
        with st.expander("##### Prediction Results"):
            model = load(model_folder+model_filename)
            input_data = pd.DataFrame([user_inputs])
            if task=="classification":
                predictions = pc.predict_model(model,data=input_data) 
                st.write(predictions)
            elif  task=="regression":   
                predictions = pr.predict_model(model,data=input_data) 
                st.write(predictions)
            else:
                st.write("Unsupported task type. Please check the target column data type.")
except:
    st.write("Please upload a file to start the analysis")