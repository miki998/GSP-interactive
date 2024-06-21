from .utils import *
from .numerical_handler import graph_analysis
import streamlit as st

def ui_box(G, A, signal):
    #Create three input widgets that allow users to specify their preferred layout and color schemes
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        layout = st.selectbox('Choose a network layout', ('Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
    with col2:
        basis = st.selectbox('Choose Basis Projection', ('GFT-Laplacian','GFT-Adjacency','MyBasis','Polar-Decomposition_in', 'Polar-Decomposition_out', 'Polar-Decomposition_inflow'))
        U, coefs, V = graph_analysis(A, signal, type=basis)
    with col3:
        eigenmode_idx = st.selectbox('Choose which Eigenmode', [f'{k+1}-th' for k in range(A.shape[0])])
        eigenmode_idx = int(eigenmode_idx[:-3])

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

    return pos, U, coefs, V, eigenmode_idx