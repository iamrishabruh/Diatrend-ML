# utils/visualization.py
import plotly.express as px
import pandas as pd

def plot_glucose_series(cgm_series: pd.Series, title="CGM Glucose Time Series"):
    df = pd.DataFrame({"Time": range(len(cgm_series)), "Glucose": cgm_series})
    fig = px.line(df, x="Time", y="Glucose", title=title)
    return fig

def plot_feature_distribution(features: dict, title="Feature Distribution"):
    df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
    fig = px.bar(df, x="Feature", y="Value", title=title)
    return fig
