import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


def knn_param_selector():

    n_neighbors = st.number_input("K_neighbors", 2, 20, 5, 1)
    metric = st.selectbox(
        "metric", ("minkowski", "euclidean", "manhattan", "chebyshev", "mahalanobis")
    )

    params = {"n_neighbors": n_neighbors, "metric": metric}

    model = KNeighborsClassifier(**params)
    return model
