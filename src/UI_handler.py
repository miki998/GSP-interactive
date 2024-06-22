from .processes import *
from .numerical_handler import graph_analysis
import streamlit as st

def ui_box(G, A, signal, pos, prepare_trace):
    #Create three input widgets that allow users to specify their preferred layout and color schemes
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        layout = st.selectbox('Choose a network layout', ('Default Layout', 'Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
        #Get the position of each node depending on the user' choice of layout
        if (len(signal.shape) > 1) and (layout != 'Default Layout'):
            st.warning('Beware! Proposed Layout here does not apply for 3D data', icon="⚠️")

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
        #Add positions of nodes to the graph
        for n, p in pos.items():
            G.nodes[n]['pos'] = p
        edge_trace_template, node_trace_template, quiver_template = prepare_trace(G, np.zeros_like(A[0]), False)

    with col2:
        basis = st.selectbox('Choose Basis Projection', ('GFT-Laplacian','GFT-Adjacency','MyBasis','Polar-Decomposition_in', 'Polar-Decomposition_out', 'Polar-Decomposition_inflow'))
        U, coefs, V = graph_analysis(A, signal, type=basis)
    with col3:
        eigenmode_idx = st.selectbox('Choose which Eigenmode', [f'{k+1}-th' for k in range(A.shape[0])])
        eigenmode_idx = int(eigenmode_idx[:-3])

    # Compute frequencies and associated magnitudes
    if len(signal.shape) > 1:
        freqs = []
        mags = []
        real_freqs = np.abs(V.imag) < 1e-10
        for n in range(len(coefs)):
            coef = coefs[n]
            freq = []
            mag = []
            for freqnb in range(len(coef)):
                # Add both the positive part and the negative part
                if (real_freqs[freqnb] == 1):
                    freq.append(-np.abs(V[freqnb]))
                    freq.append(np.abs(V[freqnb]))
                    mag.append(np.abs(coef[freqnb]))
                    mag.append(np.abs(coef[freqnb]))
                else:
                    freq.append(np.sign(V.imag[freqnb]) * np.abs(V[freqnb]))
                    mag.append(np.abs(coef[freqnb]))

            freqs.append(freq)
            mags.append(mag)
        freqs = np.asarray(freqs)
        mags = np.asarray(mags)
    else:
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
            else:
                freqs.append(np.sign(V.imag[freqnb]) * np.abs(V[freqnb]))
                mags.append(np.abs(coefs[freqnb]))
        freqs = np.asarray(freqs)
        mags = np.asarray(mags)
    
    return G, U, mags, freqs, eigenmode_idx, edge_trace_template, node_trace_template, quiver_template