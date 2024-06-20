import streamlit as st
from utils.functions import (
    add_polynomial_features,
    generate_data,
    get_model_tips,
    plot_scatter_and_metrics,
    plot_decision_boundary_and_metrics,
    plot_confusion_matrix,
    train_model,
    plot_metrics,
)

from utils.ui import (
    dataset_selector,
    polynomial_degree_selector,
    introduction,
    model_selector,
    metrics,
    display_metrics
)

st.set_page_config(
    page_title="Analyix", layout="wide", page_icon="./images/flask.png"
)

def sidebar_controllers():
    dataset, n_samples, train_split, test_split,train_noise, test_noise, n_classes, select_features, selected_label = dataset_selector()
    model_type, model = model_selector()
    x_train, y_train, x_test, y_test = generate_data(
        dataset, n_samples, train_split, test_split,train_noise, test_noise, n_classes, select_features, selected_label
    )
    st.sidebar.header("Feature engineering")
    degree = polynomial_degree_selector()
    metric = metrics()

    return (
        dataset,
        n_classes,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        degree,
        train_split, 
        test_split,
        train_noise,
        test_noise,
        n_samples,
        metric
    )
      
def body(
    x_train, x_test, y_train, y_test, degree, model, model_type, metric
):
    introduction()
    col1, col2 = st.columns((1, 1))

    with col1:
        plot_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    with col2:
        result_header_placeholder = st.empty()
        metric_placeholder = st.empty()
        result_placeholder = st.empty()

    x_train, x_test = add_polynomial_features(x_train, x_test, degree)

    # Render Metric Result
    (
        model, 
        train_accuracy, 
        train_f1,
        precision_train, 
        recall_train,  
        test_accuracy, 
        test_f1, 
        precision_test, 
        recall_test, 
        duration
    ) = train_model(model, x_train, y_train, x_test, y_test)

    # Make object of metric
    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "precision_test": precision_test,
        "recall_test": recall_test,
    }
    
    # Decision Boundary Placeholder
    # fig = plot_decision_boundary_and_metrics(
    #     model, x_train, y_train, x_test, y_test, metrics
    # )
    # fig = plot_scatter_and_metrics(model, x_train, y_train, x_test, y_test, metrics)
    
    # Confusion Matrix Placeholder
    with plot_placeholder.container():
        plot_confusion_matrix(model, x_train, y_train, x_test, y_test)
        
    # Result Placeholder
    result_header_placeholder.header(f"""
        **Result for {model_type}**
        -----
                """)
    
    # Metric Plot
    with metric_placeholder.container():
        plot_metrics(metric, model,  x_train, y_train, x_test, y_test)
    
    displayed_metrics = display_metrics(metrics)
    with result_placeholder.container():
        displayed_metrics()
        
    # Tips Placeholder
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    model_tips = get_model_tips(model_type)
    tips_placeholder.info(model_tips)

# Render data
if __name__ == "__main__":
    (
        dataset,
        n_classes,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        degree,
        train_split, 
        test_split,
        train_noise,
        test_noise,
        n_samples,
        metric
    ) = sidebar_controllers()
    body(
        x_train,
        x_test,
        y_train,
        y_test,
        degree,
        model,
        model_type,
        metric,
    )
