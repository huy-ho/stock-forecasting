from json import load
from keras import models
import streamlit as st
from utils import *
from pandas.plotting import lag_plot
from keras.models import load_model
import pickle

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
ax.set_title('Stocks Closing Prices (2014-2021)')
ax.legend()

st.pyplot(fig)

st.header('We will explore some characteristics of these companies')

st.write('We will look at the market cap, dividend yield, employee counts')
st.dataframe(data=company_comparison)
st.dataframe(data=employees)


st.header('Using ARIMA')

nvda = pd.DataFrame(data['NVDA']['Close'])
nvda.columns = ['Close']

st.subheader('We will plot the training sets and the testing sets')

fig_nvda, ax_nvda = plt.subplots()
ax_nvda.plot(train_nvda)
ax_nvda.plot(test_nvda)
ax.set_title('Train NVDA and Test NVDA')
st.pyplot(fig_nvda)


st.header('Using Keras')

app_model = load_model('keras_model.h5')

m_loss = app_model.history['loss']

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(m_loss)
st.pyplot(fig_loss)


