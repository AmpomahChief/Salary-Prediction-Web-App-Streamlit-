# importing librabries
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import re
from PIL import Image


# first line after the importation section
st.set_page_config(page_title="Salary Prediction app", page_icon='💲', layout="centered", initial_sidebar_state = 'auto')
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Setting the page title
st.title("Salary Prediction Web Application")


# Loading Machine Learning items
# st.cache_resource
# st.cache
@st.cache_resource
# (allow_output_mutation=True)
def Load_ml_items(relative_path):
    "Load ML items to reuse them"
    with open(relative_path, 'rb' ) as file:
        loaded_object = pickle.load(file)
    return loaded_object
Loaded_object = Load_ml_items('assets/App_toolkit.pkl')


# Instantiating elements of the Machine Learning Toolkit
model,encoder, scaler, data = Loaded_object['model'], Loaded_object['encoder'], Loaded_object['scaler'], Loaded_object['data']

    
# Image for the page
image = Image.open("assets/salary.jpg")
st.image(image, width = 900)

# Creating elements of the sidebar
st.sidebar.header("This Web App is a deployment of a machine model that predicts employee Salary")
check =st.sidebar.checkbox("Column discription")

##################################################################

# # Setting up variables for input data
@st.cache_resource()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame (
        dict(
            work_year=[],
            experience_level=[],
            employment_type=[],
            job_title=[],
            employee_residence=[],
            remote_ratio=[],
            company_location=[],
            company_size=[],
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

##################################################################

# Forms to retrieve input
form = st.form(key="information", clear_on_submit=True)

with form:
    # cols = st.columns((1, 1))
    cols = st.columns(2)
    work_year = cols[0].selectbox('select sales date',options = list(data['work_year'].unique()))
    experience_level = cols[1].selectbox('Please select experience level', options = list(data['experience_level'].unique()))
    
    # Second row
    employment_type = cols[0].select_slider('select employment type', options = list(data['employment_type'].unique()))
    job_title = cols[1].selectbox('What is your job title', options = list(data['job_title'].unique()))
    
    # Third row
    employee_residence = cols[0].selectbox('Please select employee residence', options = list(data['employee_residence'].unique()))
    remote_ratio = cols[1].selectbox('Please select the remote ratio', options = list(data['remote_ratio'].unique()))
    
    # Forth row   
    company_location = cols[0].selectbox('Please select company location', options = list(data['company_location'].unique()))
    company_size = cols[1].selectbox('Please select the company_size', options = list(data['company_size'].unique()))
       
    # Submit button
    submitted = st.form_submit_button(label= "Get Prediction")
    
##############################################################################
if submitted:
    st.success('Form Recieved!', icon="✔️")  
        
    
    pd.read_csv(tmp_df_file)._append(
        dict(
            work_year=work_year,
            experience_level=experience_level,
            employment_type=employment_type,
            job_title=job_title,
            employee_residence=employee_residence,
            remote_ratio=remote_ratio,
            company_location=company_location,
            company_size=company_size,
            ),
                ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    
    st.balloons()
    
    df = pd.read_csv(tmp_df_file)
    input_df = df.copy()

######################################################################

# Scaling Numerical columns
    num_cols =['remote_ratio']
    input_df[num_cols] = Loaded_object['scaler'].transform(input_df[num_cols])

# Encoding categorical columns
    categoricals = ['experience_level',
                    'employment_type',
                    'job_title',
                    'employee_residence',
                    'company_location',
                    'company_size',
                    ]
    encoded_categoricals = encoder.transform(input_df[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    # encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    
    processed_df = df.reset_index().join(encoded_categoricals, on = 'index') # Reset index to make it a column
    processed_df.drop(columns = categoricals + ['index'], inplace = True) # Drop original categorical columns and the 'index' column
    processed_df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', "", x), inplace = True)
    
    
    # processed_df = df.join(encoded_categoricals)
    # processed_df.drop(columns=categoricals, inplace=True)



###################################################################    
    
    # Making Predictions
    prediction = model.predict(processed_df)
    df["Salary($)"] = prediction
    # df["Salary($)"] = prediction
    results = prediction[-1]
    
     # Displaying predicted results
    st.success(f"**Predicted Salary**: USD {results}")

###################################################################
   
    # def predict(X, model=model):
    #     results = model.predict(X)
    #     return results
    
    # prediction = predict(processed_df, model)
    # df['sales']= prediction 
    
##################################################################

    # Compounding all predicted results
    expander = st.expander("See all records")
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['salary']= prediction
        st.dataframe(df)
    
########################################################################

#OTHERS   

# footer
footer = st.expander("**Additional Information**")
with footer:
    footer.markdown("""
                    - Access the repository of this App [here](https://github.com/MavisAJ/Store-sales-prediction-Regression___TimeSeries-Analysis-.git).
                    - Contact me on github[here](https://AmpomahChief.github.io/).
                    """)