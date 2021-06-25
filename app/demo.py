import pandas as pd
import streamlit as st

# Set tile of app
st.sidebar.title("Do No Harm")
st.sidebar.write("Here is our important app")

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
    
    st.markdown("This is an overview of your dataset:")
    inspect_data = st.dataframe(dataset.head())

    features_list = dataset.columns.tolist()
    features_list.insert(0, "None selected")

    pred = st.sidebar.selectbox(
        "Select variable to be predicted",
        features_list
    )

    sf = st.sidebar.multiselect(
        "Select sensitive feature/s",
        features_list,
        default=features_list[1]
        # dataset.columns.to_list(),
        # default=dataset.columns.to_list()[0]
    )

    if (pred is not "None selected"):
        inspect_data.empty()
        
        st.markdown("## Is the dataset balanced in number of samples?")
        st.markdown("{Description of what this means}")
        
        balances_n_samples = []
        for f in sf:
            st.write(dataset[f].value_counts())
            balance = st.checkbox(f"Check the box if {f} is balanced", "Yes")
            balances_n_samples.append(balance)
        
        st.markdown("## Is the dataset balanced with respect to other features?")
        st.markdown("{Description of what this means}")
        
        balances_features = []
        for f in sf:
            st.write(dataset.groupby(f).mean())
            balance = st.checkbox(
                f"Check the box if {f} is balanced with respect to other features", "Yes"
            )
            balances_features.append(balance)
        
        st.write(dataset.groupby(by=sf, group_keys=True).mean())
