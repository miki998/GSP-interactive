"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


from src import *

uploaded_file = display()

#Create the network graph using networkx
if uploaded_file is not None:
    A, pos, signal = load(uploaded_file)

    if len(pos[0]) == 3:
        prepare_trace = prepare_trace3d
        _graphtype = "scene"
    else:
        prepare_trace = prepare_trace2d
        _graphtype = "xy"

    G = nx.from_numpy_array(A, create_using=nx.MultiGraph)

    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], horizontal_spacing = 0.,
                        specs=[[{"type": _graphtype, "rowspan": 2}, {"type": _graphtype}],
                               [None, {}]],
                               subplot_titles=("Original Graph", "Eigenmodes", "Signal Spectrum"))

    G, U, mags, freqs, eidx, edge_trace_template, node_trace_template, quiver_figure_template = ui_box(G, A, signal, pos, prepare_trace)
    print("Magnitudes and Frequencies + Basis are generated ...")
    

    ### EIGENMODE PLOT ###
    edge_trace_eig, node_trace_eig, quiver_figure_eig = prepare_trace(G, U[:,eidx].real, False,
                                                   deepcopy(edge_trace_template), 
                                                   deepcopy(node_trace_template),
                                                   deepcopy(quiver_figure_template))
    fig_eig = go.Figure(data=[edge_trace_eig, node_trace_eig])
    for t in fig_eig.data:
        fig.append_trace(t, row=1, col=2)
    # fig.append_trace(quiver_figure_eig.data[0], row=1, col=2)


    ### SIGNAL + SPECTRUM PLOT ###
    if len(signal.shape) == 1: # STATIC
        edge_trace, node_trace, quiver_figure = prepare_trace(G, signal, True,
                                                edge_trace=deepcopy(edge_trace_template),
                                                node_trace=deepcopy(node_trace_template),
                                                quiver_figure=deepcopy(quiver_figure_template))
        print("Traces are generated ...")
        fig_signal = go.Figure(data=[edge_trace, node_trace])
        for t in fig_signal.data:
            fig.append_trace(t, row=1, col=1)
        # fig.append_trace(quiver_figure.data[0], row=1, col=1)

        fig.add_trace(go.Bar(x=freqs, y=mags), row=2, col=2)
        fig.update_yaxes(range=[mags.min(), mags.max()], row=2, col=2)


    else: # DYNAMIC
        progress_text = "Pre-computing graphs' signals and displays ... Please wait."
        my_bar = st.progress(0, text=progress_text)
        for tidx in range(signal.shape[0]):
            edge_trace, node_trace, quiver_figure = prepare_trace(G, signal[tidx], True, 
                                                   edge_trace=deepcopy(edge_trace_eig), 
                                                   node_trace=deepcopy(node_trace_eig), 
                                                   vminmax=(signal.min(), signal.max()))
            fig_signal = go.Figure(data=[edge_trace, node_trace])
            for t in fig_signal.data:
                fig.append_trace(t, row=1, col=1)
            # fig.append_trace(quiver_figure.data[0], row=1, col=1)

            fig.add_trace(go.Bar(x=freqs[tidx], y=mags[tidx]), row=2, col=2)
            fig.update_yaxes(range=[mags.min(), mags.max()], row=2, col=2)

            my_bar.progress(tidx/signal.shape[0], text=progress_text)
        my_bar.empty()

        for n in range(5,len(fig.data)):
            fig.data[n].visible = False

        # Slider
        steps = []
        for tidx in range(signal.shape[0]):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Slider switched to timepoint: " + str(tidx)}],  # layout attribute
            )
            for n in range(3):
                step["args"][0]["visible"][n] = True  # Toggle eigenmode plot's trace to always visible

            step["args"][0]["visible"][2+3*tidx] = True  # Toggle trace associated to timepoints tidx to "visible"
            step["args"][0]["visible"][2+3*tidx+1] = True
            step["args"][0]["visible"][2+3*tidx+2] = True
            # step["args"][0]["visible"][2+3*tidx+3] = True
            steps.append(step)

        sliders = [dict(active=0, currentvalue={"prefix": "Timepoint: "},
                        pad={"t": 50}, steps=steps)]
        fig.update_layout(sliders=sliders)
        print("Traces are generated ...")



    fig.update_layout(height=700, width=800, title_text="Graph Analysis")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)