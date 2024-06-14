from pathlib import Path
from matplotlib import pyplot as plt
import base64
import time
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.datasets import make_moons, make_circles, make_blobs
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler 
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from models.utils import model_infos, model_urls


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.cache_resource()
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes, select_features, selected_label):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)
    
    elif dataset == "custom":
        df = pd.read_csv('dataset.csv')
        X = df[select_features].values
        y = df[selected_label].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=n_samples, test_size=n_samples // 2, random_state=90)

    return x_train, y_train, x_test, y_test

def plot_metrics(metrics_list, model, x_train, y_train, x_test, y_test):
            # Fit model to training data
    model.fit(x_train, y_train)
            
    if "Confusion Matrix" in metrics_list:
        # Membuat prediksi
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        # Menampilkan confusion matrix tanpa indikator di sebelah kanan
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)  # Nonaktifkan colorbar

        # Ubah warna background dan warna teks
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        plt.title('Confusion Matrix', color='white')
        plt.xlabel('Predicted label', color='white')
        plt.ylabel('True label', color='white')

        # Ubah warna ticks dan label ticks
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Ubah warna teks di dalam kotak confusion matrix
        for text in disp.text_.ravel():
            text.set_color('white')

        # Buat custom colormap
        colors = ['#008000', '#ff6347']
        cmap_name = 'custom_cmap'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        # Ubah warna kotak confusion matrix
        disp.im_.set_cmap(custom_cmap)

        st.pyplot(fig)

    if "ROC Curve" in metrics_list:
        # Calculate ROC curve
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Plot ROC curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='#ff6347', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', color='white')
        plt.ylabel('True Positive Rate', color='white')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Ubah warna ticks pada axis menjadi putih
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')

        # Ubah warna latar belakang
        plt.gca().set_facecolor('white')
        plt.gcf().patch.set_facecolor('#0e1117')

        st.pyplot(plt.gcf())
        
    if "Precision-Recall Curve" in metrics_list:
        # Calculate Precision-Recall curve
        y_prob = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, color='#ff6347', lw=2, label='Precision-Recall curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('Recall', color='white')
        plt.ylabel('Precision', color='white')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
         # Ubah warna ticks pada axis menjadi putih
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')

        # Ubah warna latar belakang
        plt.gca().set_facecolor('white')
        plt.gcf().patch.set_facecolor('#0e1117')

        
        st.pyplot(plt.gcf())
        

def plot_decision_boundary_and_metrics(
    model, x_train, y_train, x_test, y_test, metrics
):
    d = x_train.shape[1] 

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p) for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Decision Boundary", None, None),
        row_heights=[0.7, 0.30],
    )

    heatmap = go.Heatmap(
        x=xx[0],
        y=y_,
        z=Z,
        colorscale=["tomato", "rgb(27,158,119)"],
        showscale=False,
    )

    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(heatmap, row=1, col=1,).add_trace(train_data).add_trace(
        test_data
    ).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2"
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_accuracy"],
            title={"text": f"Accuracy (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_accuracy"]},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_f1"],
            title={"text": f"F1 score (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_f1"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig


def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)
    precision_train = np.round(precision_score(y_train, y_train_pred, average='weighted'), 3)
    recall_train = np.round(recall_score(y_train, y_train_pred, average='weighted'), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)
    precision_test = np.round(precision_score(y_test, y_test_pred, average='weighted'), 3)
    recall_test = np.round(recall_score(y_test, y_test_pred, average='weighted'), 3)

    return model, train_accuracy, train_f1,precision_train, recall_train,  test_accuracy, test_f1, precision_test, recall_test, duration


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test