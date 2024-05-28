"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
TBMODIFIED but originally written by 
->  https://towardsdatascience.com/build-a-simple-network-graph-app-using-streamlit-e6d65efbae88
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from  PIL import Image

#Add a logo (optional) in the sidebar
logo = Image.open(r'/Users/mikichan/Desktop/logo_gsp.png')
st.sidebar.image(logo,  width=120)

#Add the expander to provide some information about the app
with st.sidebar.expander("About the App"):
     st.write("""
        This network graph app was built by My Data Talk using Streamlit and Plotly. You can use the app to quickly generate an interactive network graph with different layout choices.
     """)

#Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader('',type=['csv']) #Only accepts csv file format

#Add an app title. Use css to style the title
st.markdown(""" <style> .font {                                          
    font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Upload your data and generate an interactive network graph instantly...</p>', unsafe_allow_html=True)

#Create the network graph using networkx
if uploaded_file is not None:     
    df=pd.read_csv(uploaded_file)  
    A = list(df["Source"].unique())
    B = list(df["Target"].unique())
    node_list = set(A+B)
    G = nx.Graph() #Use the Graph API to create an empty network graph object
    
    #Add nodes and edges to the graph object
    for i in node_list:
        G.add_node(i)
    for i,j in df.iterrows():
        G.add_edges_from([(j["Source"],j["Target"])])
 
    #Create three input widgets that allow users to specify their preferred layout and color schemes
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        layout= st.selectbox('Choose a network layout',('Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
    with col2:
        color=st.selectbox('Choose color of the nodes', ('Blue','Red','Green','Orange','Red-Blue','Yellow-Green-Blue'))      
    with col3:
        title=st.text_input('Add a chart title')

    #Get the position of each node depending on the user' choice of layout
    if layout=='Random Layout':
        pos = nx.random_layout(G) 
    elif layout=='Spring Layout':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif  layout=='Shell Layout':
        pos = nx.shell_layout(G)            
    elif  layout=='Kamada Kawai Layout':
        pos = nx.kamada_kawai_layout(G) 
    elif  layout=='Spectral Layout':
        pos = nx.spectral_layout(G) 

    #Use different color schemes for the node colors depending on the user input
    if color=='Blue':
        colorscale='blues'    
    elif color=='Red':
        colorscale='reds'
    elif color=='Green':
        colorscale='greens'
    elif color=='Orange':
        colorscale='orange'
    elif color=='Red-Blue':
        colorscale='rdbu'
    elif color=='Yellow-Green-Blue':
        colorscale='YlGnBu'

    #Add positions of nodes to the graph
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    #Use plotly to visualize the network graph created using NetworkX
    #Adding edges to plotly scatter plot and specify mode='lines'
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    #Adding nodes to plotly scatter plot
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=colorscale, #The color scheme of the nodes will be dependent on the user's input
            color=[],
            size=20,
            colorbar=dict(
                thickness=10,
                title='# Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])]) #Coloring each node based on the number of connections
        node_info = str(adjacencies[0]) +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
    
    #Plot the final figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title, #title takes input from the user
                    title_x=0.45,
                    titlefont=dict(size=25),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit