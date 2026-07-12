import pandas as pd
import streamlit as st

if __name__ == "__main__":
    filename = 'output/og_kush_top_500_profit.csv'

    top_states = pd.read_csv(filename, index_col=0)
    st.dataframe(top_states)