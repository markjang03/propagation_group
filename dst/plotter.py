import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import plotly.express as px
import streamlit as st
import altair as alt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

'''
def plot_graphs(env_map=None, tirem_rssi=None):
    """Plots tirem RSSI values. If None, load values."""
    # env_map = np.load('tx_powder.npz', allow_pickle=True)['map']
    # tirem_rssi = np.load('tirem_rssi.npy')
    plotter(env_map)
    plt.title('TIREM RSSI Prediction')
    plotter(tirem_rssi)
    contour_plot(tirem_rssi)


def contour_plot(tirem_rssi):
    lvls = np.linspace(-200, -40, 10)
    x = np.arange(tirem_rssi.shape[0])
    y = np.arange(tirem_rssi.shape[1])
    plt.contourf(x, y, tirem_rssi, levels=lvls, cmap='hot')
    plt.axis('scaled')
    plt.grid(which='minor')
    plt.colorbar()
    plt.show()
'''

def plotter(value_map, title):
    value_map = value_map[::3, ::3]
    fig = px.imshow(value_map[1:value_map.shape[0] - 1, 1:value_map.shape[1] - 1], origin='lower',title=title)
    fig.update_layout(coloraxis_colorbar_x=0.75)
    fig.update_layout(coloraxis_colorbar_y=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig)


def plotter2(value_map, x_list, y_list, UTM_long, UTM_lat, title):
    #value_map = value_map[::3, ::3]
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    ax.imshow(value_map[1:value_map.shape[0] - 1, 1:value_map.shape[1] - 1], origin='lower')
    x_tick_labels = ["{:.5e}".format(val) for val in UTM_lat[:, 0][:, 0]]
    ax.set_xticks(range(0, len(UTM_lat[:, 0][:, 0]), 1000))  # You might want to adjust the ticks
    ax.set_xticklabels(x_tick_labels[0:len(UTM_lat[:, 0][:, 0]):1000], fontsize=4)

    y_tick_labels = ["{:.5e}".format(val) for val in UTM_lat[:, 0][:, 0]]
    ax.set_yticks(range(0, len(UTM_long[:, 0][:, 0]), 1000))  # You might want to adjust the ticks
    ax.set_yticklabels(x_tick_labels[0:len(UTM_long[:, 0][:, 0]):1000], fontsize=4)

    #ax.set_xticklabels(y_tick_labels, fontsize=4)
    #ax.set_yticklabels("{:.3g}".format(UTM_long[:, 0][:, 0]), fontsize=4)
    ax.set_xlabel("UTM_E [m]", fontsize=4)
    ax.set_ylabel("UTM_N [m]", fontsize=4)
    ax.set_title(title)
    ax.scatter(x_list, y_list, color="red", s=3)
    st.pyplot(fig, use_container_width=False)


def plotter3(value_map, x_list, y_list, UTM_long, UTM_lat, end_x_list, end_y_list, end_names, title, downsample_factor=1):
    value_map = value_map[::downsample_factor, ::downsample_factor]
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    fig = px.imshow(value_map[1:value_map.shape[0] - 1, 1:value_map.shape[1] - 1], origin='lower', title=title)
    fig.update_layout(coloraxis_colorbar_x=0.85)
    fig.update_layout(coloraxis_colorbar_y=0.45)
    fig.update_layout(height=500)
    #fig = px.imshow(data)

    # Create a scatter plot
    #fig2 = px.scatter(x=[x/downsample_factor for x in x_list], y=[x/downsample_factor for x in y_list], opacity=0.7,  color_discrete_sequence=['green'])
    fig2 = go.Figure(go.Scatter(
        x=[x/downsample_factor for x in x_list],
        y=[x/downsample_factor for x in y_list],
        mode='markers',
        marker=dict(
            size=3,
            color="green"
        ),
        name = "Data Collection Locations"
    ))
    #fig2.update_traces(marker={'size': 3})

    scatter_data2 = {
        'x': [int(x/downsample_factor) for x in end_x_list],
        'y': [int(y/downsample_factor) for y in end_y_list],
        #'value': [0.1, 0.5, 0.9],
        'legend_label': end_names
    }
    scatter_fig2 = go.Figure(go.Scatter(
        x=scatter_data2['x'],
        y=scatter_data2['y'],
        mode='markers',
        marker=dict(
            size=18,
            symbol = "star"
        ),
        text=scatter_data2['legend_label'],
        name = "Endpoints"
    ))

    # Overlay the scatter plot on top of the 2D array image
    fig.add_trace(fig2.data[0])
    fig.add_trace(scatter_fig2.data[0])

    # Streamlit component to display the combined plot
    st.plotly_chart(fig, use_container_width=True)
