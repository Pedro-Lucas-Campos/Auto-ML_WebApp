import streamlit as st
import pandas as pd



st.set_page_config(page_title='Data',layout='wide',
                   initial_sidebar_state='collapsed',menu_items={"Get help":"mailto:predograciano@gmail.com",
                                                                 "Report a bug": "mailto:predograciano@gmail.com",
                                                                 "About":"Conheça meu trabalho: https://linkedin.com.br/in/pedrolucasgracianocampos"})
st.title("AutoML",text_alignment='center')
st.subheader('Load your dataset:',divider='red')
dataset = st.file_uploader("",accept_multiple_files=False,)
if dataset:
    st.subheader('Archive format:',divider='red')
    file_type = st.radio('', options=['Excel','CSV','Parquet'],label_visibility='collapsed',horizontal=True)
    st.subheader('Preview your dataset:',divider='red')
    try:
        if file_type == 'CSV':
            df = pd.read_csv(dataset,index_col=0)
    except:
         st.write("THIS ARCHIVE IS NOT A CSV FILE")
    try:
         
        if file_type == "Excel":
            df = pd.read_excel(dataset,index_col=0,engine='openpyxl')
    except:
         st.write("THIS ARCHIVE IS NOT AN EXCEL FILE")
    try:     
        if file_type == "Parquet":
            df = pd.read_parquet(dataset,engine='pyarrow')
    except:
         st.write("THIS ARCHIVE IS NOT A PARQUET FILE")
    if 'df' not in st.session_state:
            st.session_state['df'] = df
    st.dataframe(df.head(6))
    start_ML = st.button("Start Machine Learning Pipeline",use_container_width=True,type='primary')
    if start_ML:
        st.switch_page(r"pages/ml.py")