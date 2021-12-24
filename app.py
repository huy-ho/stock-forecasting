import streamlit as st
from model import *

st.title('Stock Forecasting')

st.write(
    'I will be exploring Nvidia, AMD, and Intel stocks and perform analysis on them'
)

#dataframe 
st.dataframe(data=data)


#plotting the 3 stocks
arr_nvda = data['NVDA']['Close']
arr_amd = data['AMD']['Close']
arr_intc = data['INTC']['Close']

fig, ax = plt.subplots()
ax.plot(arr_nvda, color='green', label='NVDA')
ax.plot(arr_amd, color='red', label='AMD')
ax.plot(arr_intc, color='blue', label='INTC')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Prices')
ax.legend()

st.pyplot(fig)