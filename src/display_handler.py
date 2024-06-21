from .utils import *
import streamlit as st

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
            This network graph app was built by My Data Talk using Streamlit and Plotly. You can use the app to quickly generate an interactive network graph with different layout choices.
        """)

    #Add a file uploader to the sidebar
    uploaded_file = st.sidebar.file_uploader('',type=['pkl']) #Only accepts csv file format
    uploaded_file = './resources/demo_data/usa_graph/interactivepack_usa.pkl'

    #Add an app title. Use css to style the title
    st.markdown(""" <style> .font {                                          
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Interactive GSP - #In a Nutshell</p>', unsafe_allow_html=True)

    return uploaded_file