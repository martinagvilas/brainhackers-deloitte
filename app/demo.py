import pandas as pd
import streamlit as st

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Important functions
def compute_t_test(df, sensitive_feature):
    
    significant_features = []
    for feature in df.columns:
        t_val, p_val = stats.ttest_ind(
            df.loc[0][feature].values, 
            df.loc[1][feature].values
        )
        
        print('done')
        if p_val < 0.05:
            significant_features.append(feature)
    
    return significant_features


def run_ml_model(df, pred):
    
    # Create X and y
    X = df.copy().drop([pred], axis=1)
    y = df.copy()[pred].values
    
    # Create preprocessor of features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())]
    )

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create pipeline
    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ]
    )

    # Split into train and dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    # Train classifier
    clf = clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test 


# Set configuration
st.set_page_config(layout="wide")

# Set tile of app
st.sidebar.title("Do No Harm")
st.sidebar.write("Here is our important app")

# Load dataset
@st.cache
def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    return dataset

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    dataset = load_dataset(uploaded_file)
    
    # Set types
    categorical_features = ["sex", "rent", "minority", "ZIP", "occupation"]
    numeric_features = [
        "education", "age", "income", "loan_size", "payment_timing",
        "year", "job_stability"
    ]
    for cat in categorical_features:
        dataset[cat] = dataset[cat].astype("object")
    
    # Print data
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
        st.title("Data Bias")
        
        st.markdown("## How do sensitive features distribute across the outcome variable?")
        st.markdown("{Description of what this meansk}")
        
        balances_dist = []
        for f in sf:
            grouped_dataset = dataset.copy().groupby([pred, f]).size()
            #grouped_dataset.rename(columns={0: "Count"})
            st.write(grouped_dataset)
            balance_dist = st.checkbox(f"Check the box if {f} is balanced n", "Yes")
            balances_dist.append(balance_dist)
        
        st.markdown("## Is the dataset balanced in the number of samples on each sensitive feature?")
        st.markdown("{Description of what this meansd}")
        
        balances_n_samples = []
        for f in sf:
            st.write(dataset[f].value_counts())
            balance = st.checkbox(f"Check the box if {f} is balanced f", "Yes")
            balances_n_samples.append(balance)
        
        st.markdown("## Is the dataset balanced with respect to other features?")
        st.markdown("{Description of what this means}")
        
        balances_features = []
        for f in sf:
            grouped_dataset = dataset.copy().set_index(f)
            
            categorical_dataset = (
                grouped_dataset.select_dtypes(include=["object", "bool"])
            )
            categorical_dataset = pd.get_dummies(categorical_dataset)
            st.write(categorical_dataset.groupby(level=0).count())
            balance = st.checkbox(
                f"Check the box if {f} is balanced with respect to other features", "Yes"
            )
            balances_features.append(balance)

            numerical_dataset = (
                grouped_dataset.select_dtypes(include=["number"])
            )
            significant_features = compute_t_test(numerical_dataset, f)
            
            st.write(numerical_dataset.groupby(f).mean())
            st.markdown(f"{significant_features}")
        
        st.title("Model Bias")

        # Train model
        clf, X_train, X_test, y_train, y_test = run_ml_model(dataset, pred)
        
        y_pred = clf.predict(X_train)
        st.write(y_pred)