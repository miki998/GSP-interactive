from .processes import *
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff

def display():
    st.set_page_config(page_title="", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 300px;
            margin-left: -400px;
        }
        
        """,
        unsafe_allow_html=True,
        )    
    #Add a logo (optional) in the sidebar
    logo = Image.open(r'./resources/logo_gsp.png')
    st.sidebar.image(logo,  width=300)

    #Add the expander to provide some information about the app
    with st.sidebar.expander("About the App"):
        st.write("""
            The purpose of this app is to bring together perspectives of different basis (variants of GFT depending on use cases) in an interactive fashion.
            Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
        """)

    #Add a file uploader to the sidebar
    uploaded_file = st.sidebar.file_uploader('Option 1',type=['pkl']) #Only accepts csv file format
    # uploaded_file = './resources/demo_data/macaque_data/interactivepack_macaque_dynamic.pkl'
    # uploaded_file = './resources/demo_data/macaque_data/interactivepack_macaque_static.pkl'
    # uploaded_file = './resources/demo_data/usa_graph/interactivepack_usa_static.pkl'
    # uploaded_file = './resources/demo_data/usa_graph/interactivepack_usa_dynamic.pkl'
    
    graphtype = st.sidebar.selectbox('Option 2: Choose a demo graph to start!', ('', 'USA', 'Duo-Modular','Cycle','Brain'))
    if graphtype == 'USA':
        uploaded_file = './resources/demo_data/usa_graph/interactivepack_usa_dynamic.pkl'
    elif graphtype == 'Brain':
        uploaded_file = './resources/demo_data/macaque_data/interactivepack_macaque_dynamic.pkl'
    else:
        uploaded_file = None
    #Add an app title. Use css to style the title
    st.markdown(""" <style> .font {                                          
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Interactive GSP - #In a Nutshell</p>', unsafe_allow_html=True)

    return uploaded_file

def prepare_trace2d(G, signal, colorbarflag=False, edge_trace=None, node_trace=None, quiver_figure=None, vminmax=None):
    # Use plotly to visualize the network graph created using NetworkX
    # Adding edges to plotly scatter plot and specify mode='lines'
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1,color='#888'), 
                            hoverinfo='none', mode='lines')
    quiver_x = []
    quiver_y = []
    quiver_u = []
    quiver_v = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']

        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

        quiver_x.append(x0)
        quiver_y.append(y0)
        quiver_u.append(x1-x0)
        quiver_v.append(y1-y0)

    quiver_figure = ff.create_quiver(np.array(quiver_x), np.array(quiver_y),
                                      np.array(quiver_u), np.array(quiver_v))
    #Adding nodes to plotly scatter plot
    if colorbarflag:
        node_trace = go.Scatter(x=[],y=[],text=[],mode='markers',hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Bluered', #The color scheme of the nodes will be dependent on the user's input
                color=[],
                size=20,
                colorbar=dict(
                    thickness=10,
                    title='# Signal Strength',
                    xanchor='left',
                    titleside='top'
                ), colorbar_x=0,
                line=dict(width=0)))
    else:
        node_trace = go.Scatter(x=[],y=[],text=[],mode='markers',hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='Bluered', #The color scheme of the nodes will be dependent on the user's input
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


    if colorbarflag:
        node_trace['marker']['showscale'] = True
        node_trace['marker']['size'] = 20

    node_trace['marker']['color'] = []
    node_trace['text'] = []
    if not (vminmax is None):
        node_trace['marker']['cmin'] = vminmax[0]
        node_trace['marker']['cmax'] = vminmax[1]

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([signal[adjacencies[0]]]) #Coloring each node based on the number of connections
        node_info = 'node ' + str(adjacencies[0]) +' signal = '+str(np.round(signal[adjacencies[0]], 3))
        node_trace['text']+=tuple([node_info])

    return edge_trace, node_trace, quiver_figure


def prepare_trace3d(G, signal, colorbarflag=False, edge_trace=None, node_trace=None, quiver_figure=None, vminmax=None):
    if edge_trace is None:
        edge_trace = go.Scatter3d(x=[], y=[], z=[], line=dict(width=1,color='#888'), 
                                hoverinfo='none', mode='lines')
        progress_text = "Displaying Edges ... Please wait."
        my_bar = st.progress(0, text=progress_text)
        for edge_idx, edge in enumerate(G.edges()):
            x0, y0, z0 = G.nodes[edge[0]]['pos']
            x1, y1, z1 = G.nodes[edge[1]]['pos']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            edge_trace['z'] += tuple([z0, z1, None])

            my_bar.progress(edge_idx/len(G.edges()), text=progress_text)
        my_bar.empty()

    if node_trace is None:
        #Adding nodes to plotly scatter plot
        if colorbarflag:
            node_trace = go.Scatter3d(x=[],y=[],z=[],text=[],mode='markers',hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Bluered', #The color scheme of the nodes will be dependent on the user's input
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=10,
                        title='# Signal Strength',
                        xanchor='left',
                        titleside='top'
                    ), colorbar_x=0,
                    line=dict(width=0)))
        else:
            node_trace = go.Scatter3d(x=[],y=[],z=[],text=[],mode='markers',hoverinfo='text',
                marker=dict(
                    showscale=False,
                    colorscale='Bluered', #The color scheme of the nodes will be dependent on the user's input
                    color=[],
                    size=20,
                    colorbar=dict(
                        thickness=10,
                        title='# Signal Strength',
                        xanchor='left',
                        titleside='top'
                    ), colorbar_x=0,
                    line=dict(width=0)))


        for node in G.nodes():
            x, y, z = G.nodes[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['z'] += tuple([z])

    if colorbarflag:
        node_trace['marker']['showscale'] = True
        node_trace['marker']['size'] = 20

    node_trace['marker']['color'] = []
    node_trace['text'] = []
    if not (vminmax is None):
        node_trace['marker']['cmin'] = vminmax[0]
        node_trace['marker']['cmax'] = vminmax[1]
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([signal[adjacencies[0]]]) #Coloring each node based on signal
        node_info = 'node ' + str(adjacencies[0]) +' signal = '+str(np.round(signal[adjacencies[0]], 3))
        node_trace['text'] += tuple([node_info])

    return edge_trace, node_trace, None