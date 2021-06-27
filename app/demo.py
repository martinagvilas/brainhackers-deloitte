import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from scipy import stats
from fairlearn.metrics import MetricFrame
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Functions
def compute_t_test(df, sensitive_feature):
    
    indexes = list(set(df.index.get_level_values(0)))

    significant_features = []
    for feature in df.columns:
        t_val, p_val = stats.ttest_ind(
            df.loc[indexes[0]][feature].values, 
            df.loc[indexes[1]][feature].values
        )

        if p_val < 0.05:
            significant_features.append(feature)
    
    return significant_features

def highlight_col(x):
    r = 'background-color: red'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.loc[:, 0] = r
    return df1    

@st.cache
def run_ml_model(df, pred):
    
    # Create X and y
    X = df.copy().drop([pred], axis=1)
    y = (dataset.copy()[pred] != f"{pred}-no").astype(int).values
    
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

    return clf, X, y, X_train, X_test, y_train, y_test 


@st.cache
def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    return dataset


# Set configuration
#st.set_page_config(layout="wide")

# Set title of app
st.sidebar.image("app/deloitte_pitch_logo_square_blackV2.png", width=180, clamp=True)

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
        
        st.markdown("## Is the dataset __balanced__?")
        st.markdown(
            "### Is the __number of occurances__ balanced for each\
            __sensitive feature__?"
        )
        
        balances_n_samples = []
        for f in sf:
            st.write(f"Number of occurrances per `{f}` value:")
            balance_samples_df = dataset[f].copy().value_counts()
            balance_samples_df.name = "number"
            st.dataframe(balance_samples_df)
            #balance = st.checkbox(f"Check the box if {f} is balanced f", "Yes")
            #balances_n_samples.append(balance)
        
        st.markdown(
            "### Is the __number of occurances__ balanced for each\
            __sensitive feature__ and __outcome variable__ value?"
        )
        
        balances_dist = []
        for f in sf:
            st.write(f"Number of occurrances per `{f}` and `{pred}` value:")
            balance_outcome_df = dataset.copy().groupby([f, pred]).size()
            balance_outcome_df.name = "number"
            st.write(balance_outcome_df)
            #balance_dist = st.checkbox(f"Check the box if {f} is balanced n", "Yes")
            #balances_dist.append(balance_dist)
        

        st.markdown(
            "## How do __sensitive features relate__ to __other features__?"
        )
        
        balances_features = []
        for f in sf:
            st.markdown(
                f"How does `{f}` relate to other __categorical features__?"
            )
            grouped_dataset = dataset.copy().set_index(f)
            categorical_dataset = (
                grouped_dataset.copy().select_dtypes(include=["object", "bool"])
            )
            categorical_dataset = pd.get_dummies(categorical_dataset)
            st.write(categorical_dataset.groupby(level=0).count())
            #balance = st.checkbox(
            #    f"Check the box if {f} is balanced with respect to other features", "Yes"
            #)
            #balances_features.append(balance)

            st.markdown(
                f"How does `{f}` relate to other __continuous features__?"
            )
            numeric_dataset = (
                grouped_dataset.select_dtypes(include=["number"])
            )
            significant_features = compute_t_test(numeric_dataset, f)
            
            sign_numeric_df = numeric_dataset.groupby(f).mean()
            sign_numeric_df = (
                sign_numeric_df.style.set_properties(**{'background-color': 'yellow'}, subset=significant_features)
            )
            st.dataframe(sign_numeric_df)
            #st.write(f"{significant_features}")
            
            for sig_f in significant_features:
                st.markdown(
                    f"The subgroups of `{f}` differ significantly in `{sig_f}`:"
                )
                fig, ax = plt.subplots()
                sns.stripplot(
                    x=f, y=sig_f, hue=f, data=dataset, 
                    alpha=0.1, palette="Set2",
                    ax=ax
                )
                sns.despine()
                ax.get_legend().remove()
                plt.subplots_adjust(
                    left=0.1, right=0.9, bottom=0.2, top=0.8
                )
                st.pyplot(fig)

        
        st.title("Model Bias")

        # Train model
        clf, X, y, X_train, X_test, y_train, y_test = run_ml_model(dataset, pred)
        
        # Predict labels
        y_pred = clf.predict(X)
        
        # Calculate allocation harm
        st.markdown("## Do we observe __allocation harm__?")
        st.markdown(
            "__Allocation harms__ may occur when AI models assign or withhold\
            opportunities, resources, or information to certain subpopulations\
            (for more information see the \
            [fairlearn](https://github.com/fairlearn/fairlearn) toolbox).")
        
        for f in sf:
            pred_grouped = pd.DataFrame(
                {f"{f}": dataset[f], "y_pred": y_pred, "y_true": y}
            )
            pred_vals = (
                pred_grouped.groupby(f).sum().values 
                / dataset[f].value_counts().values
            )
            pred_grouped = pd.DataFrame(
                pred_vals, columns=[f"{pred}_predicted", f"{pred}_true"]
            )
            
            st.markdown(f"The predicted and true `{pred}` for `{f}` is:")
            st.dataframe(pred_grouped)
        
        # Calculate quality of service harm
        acc = np.round(clf.score(X_test, y_test), 3)

        grouped_metric = MetricFrame(
            recall_score, y, y_pred,
            sensitive_features=dataset[sf]
        )

        st.markdown("## Do we observe __quality of service harm__?")
        st.markdown(
            "__Quality of service harms__ may ocurr when the AI model performs\
            better for certain subpopulations (for more information see the \
            [fairlearn](https://github.com/fairlearn/fairlearn) toolbox)."
        )
        st.markdown(
            f"The __overall predictive mean accuracy__ of the model is \
            __{acc}__. But what is the accuracy for each subgroup?"
        )
        
        for f in sf:
            grouped_metric = MetricFrame(
                {"precision": precision_score, "recall": recall_score}, 
                y, y_pred,
                sensitive_features=dataset[f]
            )

            st.markdown(f"The performance of the model breaked by `{f}` is:")
            grouped_df = grouped_metric.by_group
            st.dataframe(grouped_df.astype(float).round(4))

        
