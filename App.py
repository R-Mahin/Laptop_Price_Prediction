import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load the trained model
#with open(r"F:\ML2\price_pred.pkl", "rb") as file:
#   model = pickle.load(file)

file_path = os.path.join(os.path.dirname(__file__), "price_pred.pkl")
with open(file_path, "rb") as file:
    model = pickle.load(file)

st.title("💻 Laptop Price Predictor")


# Company names
company_names = ['Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Others', 'Toshiba']
company_dict = {i: 0 for i in company_names}
comp_selected = st.selectbox('Select a Company:', company_names)
company_dict[comp_selected] = 1

# CPU company
cpu_company_names = ['Intel', 'Samsung']
cpu_company_dict = {i: 0 for i in cpu_company_names}
cpu_comp_selected = st.radio('Select a CPU:', cpu_company_names)
cpu_company_dict[cpu_comp_selected] = 1

# CPU series
cpu_series_names = ['i3', 'i5', 'i7']
cpu_series_dict = {i: 0 for i in cpu_series_names}
series_selected = st.radio('Select a CPU Series:', cpu_series_names)
cpu_series_dict[series_selected] = 1

# GPU names
gpu_names = ['ARM', 'Intel', 'Nvidia']
gpu_dict = {i: 0 for i in gpu_names}
gpu_selected = st.radio('Select a GPU:', gpu_names)
gpu_dict[gpu_selected] = 1

# Type names
type_names = ['Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
type_dict = {i: 0 for i in type_names}
type_selected = st.selectbox('Select a Laptop Type:', type_names)
type_dict[type_selected] = 1

# Width selection
width = st.selectbox('Select a Width:', [1366, 1440, 1600, 1920, 2160, 2256, 2304, 2400, 2560, 2736, 2880, 3200, 3840])

# Input fields
Ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2)
SSD = st.number_input("SSD Storage (GB)", min_value=0, max_value=2000, step=128)
HDD = st.number_input("HDD Storage (GB)", min_value=0, max_value=50, step=1)

# Button for prediction
if st.button("Predict Price"):
    # Create DataFrame
    df_fet = {
        'Ram(GB)': Ram, 'SSD': SSD, 'HDD': HDD,
        'cpu_com_Intel': cpu_company_dict["Intel"], 'cpu_com_Samsung': cpu_company_dict["Samsung"],
        'CPU_Series_i3': cpu_series_dict['i3'], 'CPU_Series_i5': cpu_series_dict['i5'], 'CPU_Series_i7': cpu_series_dict['i7'],
        'Gpu_ARM': gpu_dict['ARM'], 'Gpu_Intel': gpu_dict['Intel'], 'Gpu_Nvidia': gpu_dict['Nvidia'],
        'Company_Apple': company_dict['Apple'], 'Company_Asus': company_dict['Asus'], 'Company_Dell': company_dict['Dell'],
        'Company_HP': company_dict["HP"], 'Company_Lenovo': company_dict["Lenovo"], 'Company_MSI': company_dict["MSI"],
        'Company_Other': company_dict["Others"], 'Company_Toshiba': company_dict["Toshiba"],
        'TypeName_Gaming': type_dict["Gaming"], 'TypeName_Netbook': type_dict['Netbook'], 'TypeName_Notebook': type_dict["Notebook"],
        'TypeName_Ultrabook': type_dict["Ultrabook"], 'TypeName_Workstation': type_dict["Workstation"], 'Width': width
    }

    # Convert to DataFrame & ensure correct column types
    df_fin = pd.DataFrame([df_fet]).astype(float)

    # Ensure column order matches model input
    df_fin = df_fin[model.feature_names_in_]

    # Predict price
    predicted_price = (np.exp(model.predict(df_fin))[0])*1.39

    # Display result
    st.success(f"💰 Predicted Laptop Price: {predicted_price:.2f} Taka")
