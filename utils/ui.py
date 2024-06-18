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

def introduction():
    st.title("**Welcome to Machine Learning Test🧪**")
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
    for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)
        
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    with dataset_container: 
        
        dataset = st.selectbox("Choose a dataset", ("moons", "circles", "blobs","custom"))
        
        if dataset == "custom":
            df = pd.read_csv('dataset.csv')
            column_names = df.columns.tolist()
            
            select_features = st.multiselect(
            "Select Features", 
            column_names,
            default=column_names[:]
            )
    
            selected_label = st.selectbox(
                "Select Target (Y variable)", 
                column_names, 
                index=len(column_names) - 1
            )
            
            # Set Noise to 0 
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
        
        if(dataset != "custom"):
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
