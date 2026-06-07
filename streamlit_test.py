import sys
import streamlit as st

st.write("Python:", sys.executable)

from langchain_community.vectorstores import FAISS

st.success("FAISS imported successfully")