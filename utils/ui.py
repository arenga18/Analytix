import numpy as np
import pandas as pd
import streamlit as st
from sidebar import sidebar

from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gb_param_selector
from sklearn.model_selection import train_test_split


from models.utils import model_imports
from utils.functions import img_to_bytes


def introduction():
    st.title("**Welcome to playground 🧪**")
    st.subheader(
        """
        This is a place where you can get familiar with machine learning models directly from your browser
        """
    )

    st.markdown(
        """
    - 🗂️ Choose a dataset
    - ⚙️ Pick a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics and decision boundary on train and test data
    - 🩺 Diagnose possible overitting and experiment with other settings
    -----
    """
    )


def dataset_selector():
    col1, col2 = st.columns((1, 1))

    for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)
        
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    with dataset_container: 
        
        dataset = st.selectbox("Choose a dataset", ("moons", "circles", "blobs","custom"))
        
        if dataset == "custom":
            df = pd.read_csv('dataset.csv')
            column_names = df.columns.tolist()
            number_feature = st.number_input("Number of features", 2, 2, 2)
            
            select_features = []
            
            for i in range(number_feature):
                n_feature = st.selectbox(f"Select Features {i+1}", column_names,index= i+1,key={i+1})
                select_features.append(n_feature)
            
            selected_label = st.selectbox("Select Target", column_names, index=len(column_names)-1) 
                
            train_noise = 0
            test_noise = 0
        else:
            select_features = None
            selected_label = None
        
        n_samples = st.number_input(
            "Number of samples",
            min_value=0,
            max_value=1000,
            step=10,
            value=500,
        )

        if dataset != "custom":
            train_noise = st.slider(
            "Set the noise (train data)",
            min_value=0.01,
            max_value=0.2,
            step=0.005,
            value=0.06,
        )
            test_noise = st.slider(
            "Set the noise (test data)",
            min_value=0.01,
            max_value=1.0,
            step=0.005,
            value=train_noise,
        )
        
        if dataset == "blobs":
            n_classes = st.number_input("centers", 2, 5, 2, 1)
        else:
            n_classes = None

    return dataset, n_samples, train_noise, test_noise, n_classes, select_features, selected_label


def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()

    return model_type, model


def generate_snippet(
    model, model_type, n_samples, train_noise, test_noise, dataset, degree
):
    train_noise = np.round(train_noise, 3)
    test_noise = np.round(test_noise, 3)

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if degree > 1:
        feature_engineering = f"""
    >>> for d in range(2, {degree+1}):
    >>>     x_train = np.concatenate((x_train, x_train[:, 0] ** d, x_train[:, 1] ** d))
    >>>     x_test= np.concatenate((x_test, x_test[:, 0] ** d, x_test[:, 1] ** d))
    """

    if dataset == "moons":
        dataset_import = "from sklearn.datasets import make_moons"
        train_data_def = (
            f"x_train, y_train = make_moons(n_samples={n_samples}, noise={train_noise})"
        )
        test_data_def = f"x_test, y_test = make_moons(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "circles":
        dataset_import = "from sklearn.datasets import make_circles"
        train_data_def = f"x_train, y_train = make_circles(n_samples={n_samples}, noise={train_noise})"
        test_data_def = f"x_test, y_test = make_circles(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "blobs":
        dataset_import = "from sklearn.datasets import make_blobs"
        train_data_def = f"x_train, y_train = make_blobs(n_samples={n_samples}, clusters=2, noise={train_noise* 47 + 0.57})"
        test_data_def = f"x_test, y_test = make_blobs(n_samples={n_samples // 2}, clusters=2, noise={test_noise* 47 + 0.57})"
    
    elif dataset == "custom":
        dataset_import = "pd.read_csv('dataset.csv')"
        train_data_def = f"x_train, y_train = x_train, y_train"
        test_data_def = f"x_test, y_test = x_test, y_test"

    snippet = f"""
    >>> {dataset_import}
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score

    >>> {train_data_def}
    >>> {test_data_def}
    {feature_engineering if degree > 1 else ''}    
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet    

def display_metrics(metrics):
    def render_metrics():
        col1, col2 = st.columns((1,1))
        
        with col1:
            st.write("### (Training)")
            st.write(f"Accuracy: {metrics['train_accuracy']}")
            st.write(f"Precision: {metrics['precision_train']}")
            st.write(f"Recall: {metrics['recall_train']}")
            st.write(f"F1-Score: {metrics['train_f1']}")

        with col2:
            st.write("### (Test)")
            st.write(f"Accuracy: {metrics['test_accuracy']}")
            st.write(f"Precision: {metrics['precision_test']}")
            st.write(f"Recall: {metrics['recall_test']}")
            st.write(f"F1-Score: {metrics['test_f1']}")
    
    return render_metrics

def polynomial_degree_selector():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)

def metrics():
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    return metrics
