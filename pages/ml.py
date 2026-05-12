import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt
import pandas as pd
import sklearnex
import joblib

sklearnex.patch_sklearn()

st.title("CONFIGURATION", text_alignment='center')
df = st.session_state.df

st.subheader('SELECT THE PROBLEM TYPE: ',divider='red')
problem = st.radio('SELECT',options=['Regression','Classification'],horizontal=True,captions=["Single numerical target",
                                                                                                       "Multi numerical/categorical target"],
                                                                                                       index=None,label_visibility='collapsed')
st.divider()
st.subheader('SELECT THE TARGET COLUMN:',divider='red')
target = st.selectbox(label='SELECT',options=df.columns,label_visibility='collapsed',index=None)
st.divider()
if target:
    if problem == "Classification":
        with st.status('Running',expanded=True):
            st.write("Initializing model...")
            time.sleep(1)
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import ConfusionMatrixDisplay, classification_report
            model = LogisticRegression(max_iter=2000)
            st.write('Model initialized...')
            time.sleep(1)
            X = df.drop(columns = target)
            X_encoded = OneHotEncoder(handle_unknown='ignore').fit_transform(X=X)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,shuffle=True,
                                                                random_state=42,
                                                                stratify=y,
                                                                test_size=0.3)
            
            st.write("Data Transformation...")
            model.fit(X_train,y_train)
            score = str(f'{model.score(X_test,y_test):.2%}')
            st.write(f'Accuracy score: {score}')
            col1,col2 = st.columns(2)
            with col1:
                st.subheader('CONFUSION MATRIX')
                st.pyplot(ConfusionMatrixDisplay.from_estimator(model,X=X_test,y=y_test).figure_,width=600)
                
            with col2:
                st.subheader('Classification report')
                st.dataframe(pd.DataFrame(classification_report(y_true=y_test,y_pred=model.predict(X_test),output_dict=True)))
        joblib.dump(model,'classification_model.pkl')
        @st.fragment
        def download_classification():
            with open("classification_model.pkl", "rb") as f:
                st.download_button(
                    label="Download Model",
                    data=f,
                    file_name="classification_model.pkl",
                    mime="application/octet-stream")

        def show_code_class():
            st.code(body="""
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import ConfusionMatrixDisplay, classification_report
            model = LogisticRegression(max_iter=2000)
            X = df.drop(columns = target)
            X_encoded = OneHotEncoder().fit_transform(X=X)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,shuffle=True,
                                                                random_state=42,
                                                                stratify=y,
                                                                test_size=0.3)
            
            model.fit(X_train,y_train)
            score = str(f'{model.score(X_test,y_test):.2%}')
            st.write(f'Accuracy score: {score}')
            col1,col2 = st.columns(2)
            with col1:
                st.subheader('CONFUSION MATRIX')
                st.pyplot(ConfusionMatrixDisplay.from_estimator(model,X=X_test,y=y_test).figure_,width=600)
                
            with col2:
                st.subheader('Classification report')
                st.dataframe(pd.DataFrame(classification_report(y_true=y_test,y_pred=model.predict(X_test),output_dict=True)))
                joblib.dump(model,'classification_model.pkl')""",
            language='python',line_numbers=True)


        download_classification()
        with st.expander('Show code',expanded=False):
            show_code_class()

    if problem == "Regression":
        with st.status('Running',expanded=True):
            st.write("Initializing model...")
            time.sleep(1)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(n_jobs=-1)
            st.write('Model initialized...')
            time.sleep(1)
            X = df.drop(columns = target)
            X_encoded = OneHotEncoder(handle_unknown='ignore').fit_transform(X=X)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,shuffle=True,
                                                                random_state=42,
                                                                stratify=y,
                                                                test_size=0.3)
            
            st.write("Data Transformation...")
            model.fit(X_train,y_train)
            score = str(f'{model.score(X_test,y_test):.2%}')
            st.write(f'R2 score: {score}')

        joblib.dump(model,'regression_model.pkl')
        @st.fragment
        def download_regression():
            with open("regression_model.pkl", "rb",) as f:
                st.download_button(
                    label="Download Model",
                    data=f,
                    file_name="regression_model.pkl",
                    mime="application/octet-stream"
                )

        def show_code_class():
            st.code(body="""
            time.sleep(1)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(n_jobs = -1)
            X = df.drop(columns = target)
            X_encoded = OneHotEncoder().fit_transform(X=X)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,shuffle=True,
                                                                random_state=42,
                                                                stratify=y,
                                                                test_size=0.3)
            
            model.fit(X_train,y_train)
            score = str(f'{model.score(X_test,y_test):.2%}')
            st.write(f'R2 score: {score}')
            joblib.dump(model,'regression_model.pkl')""",
            language='python',line_numbers=True)

        download_regression()
        with st.expander('Show code',expanded=False):
            show_code_class()
