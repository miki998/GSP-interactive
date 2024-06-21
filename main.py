"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
TBMODIFIED but originally written by 
->  https://towardsdatascience.com/build-a-simple-network-graph-app-using-streamlit-e6d65efbae88
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


from src import *

def prepare_trace(G, signal):
    #Use plotly to visualize the network graph created using NetworkX
    #Adding edges to plotly scatter plot and specify mode='lines'
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1,color='#888'), 
                            hoverinfo='none', mode='lines')
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    #Adding nodes to plotly scatter plot
    node_trace = go.Scatter(x=[],y=[],text=[],mode='markers',hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='thermal', #The color scheme of the nodes will be dependent on the user's input
            color=[],
            size=20,
            colorbar=dict(
                thickness=10,
                title='# Signal Strength',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([signal[adjacencies[0]]]) #Coloring each node based on the number of connections
        node_info = 'node ' + str(adjacencies[0]) +' signal = '+str(np.round(signal[adjacencies[0]], 3))
        node_trace['text']+=tuple([node_info])

    return edge_trace, node_trace

uploaded_file = display()

#Create the network graph using networkx
if uploaded_file is not None:
    A, pos, signal = load(uploaded_file)

    G = nx.from_numpy_array(A, create_using=nx.MultiGraph)

    if pos is None:
        pos, U, coefs, V, eidx = ui_box(G, A, signal)
    else:
        _, U, coefs, V, eidx = ui_box(G, A, signal)
    #Add positions of nodes to the graph
    for n, p in pos.items():
        G.nodes[n]['pos'] = p


    edge_trace1, node_trace1 = prepare_trace(G, signal)
    edge_trace2, node_trace2 = prepare_trace(G, U[:,eidx].real)

    # Arranging Figures
    fig1 = go.Figure(data=[edge_trace1, node_trace1],
                layout=go.Layout())
    
    fig2 = go.Figure(data=[edge_trace2, node_trace2],
                layout=go.Layout())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Original Graph", "Eigenmodes"))

    for t in fig1.data:
        fig.append_trace(t, row=1, col=1)
    for t in fig2.data:
        fig.append_trace(t, row=1, col=2)

    fig.update_layout(height=500, width=800, title_text="Graph Analysis")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit


    # Compute frequencies and associated magnitudes
    freqs = []
    mags = []
    real_freqs = np.abs(V.imag) < 1e-10
    for freqnb in range(len(coefs)):
        # Add both the positive part and the negative part
        if (real_freqs[freqnb] == 1):
            freqs.append(-np.abs(V[freqnb]))
            freqs.append(np.abs(V[freqnb]))
            mags.append(np.abs(coefs[freqnb]))
            mags.append(np.abs(coefs[freqnb]))
    freqs = np.asarray(freqs)
    mags = np.asarray(mags)

    df = pd.DataFrame.from_dict({'frequencies': freqs,
                                 'coefs': mags})

    fig_distrib = px.bar(df, x='frequencies', y='coefs')
    fig_distrib.update_layout(height=300, width=300, title_text="Signal Spectrum")
    st.plotly_chart(fig_distrib, use_container_width=True) #Show the graph in streamlit