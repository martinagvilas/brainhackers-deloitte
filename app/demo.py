import pandas as pd
import streamlit as st

# Set tile of app
st.title("Do Not Harm")
st.write("Here is our important app")

# Set title
#st.title("")

# Load dataset
@st.cache
def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    return dataset

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    dataset = load_dataset(uploaded_file)
    
    st.markdown("## Inspect your data")
    st.markdown("Here is an overview of your data:")
    st.write(dataset.head())

    pred = st.sidebar.selectbox(
        "Select variable to be predicted",
        dataset.columns
    )

    sf = st.sidebar.multiselect(
        "Select sensitive feature",
        dataset.columns.to_list()
    )

    st.markdown("## Is your dataset balanced?")
    
    st.write(dataset[sf].value_counts())

    balance = st.checkbox("Check the box if the dataset is balanced", "Yes")
    st.write(balance)
