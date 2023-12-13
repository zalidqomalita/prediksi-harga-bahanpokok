from dataclasses import dataclass
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.dates as dates
import time

@st.cache(suppress_st_warning=True)
def get_data_api():
    # Mapping Kode KOMODITI
    # BERAS PREMIUM = 24 Q5
    # CABE MERAH = 2 Q12
    # DAGING AYAM = 4 Q7 
    # TELUR AYAM = 5 Q25
    # DAGIG SAPI = 6 Q6
    # MINYAK GORENG = 7 Q22
    # BAWANG MERAH = Q9
    # BAWANG PUTIH = Q10
    # GULA = Q17
    resp = requests.get('https://kf.kobotoolbox.org/api/v2/assets/avMqa5nFYcGcLBaGiCWk8Z/data.json', headers ={   
            'Authorization': 'Token a3de28eb5ffbea5f319be5202b681ba3c964fb35',
            'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
            })
    resp_dict = resp.json()
    resp_dict=resp_dict['results']

    return resp_dict

def predict(X_test, model):
    pred = model.predict(X_test)
    return pred

@st.experimental_singleton(suppress_st_warning=True)
def load_model_ml(komoditas):
    if komoditas == 'Beras Premium':
        model_beras = load_model('model_prediksi_beras_premium_lstm_lag30.h5')
        model = model_beras
        print("Model  Beras Premium Loaded")
        kode = 'group_beras/Q5'
        lag = 30
    elif komoditas == 'Cabai Merah':
        model_cabai = load_model('model_prediksi_cabai_merah_lstm.h5')
        model = model_cabai
        kode = 'group_sayur/Q12'
        lag = 60
    elif komoditas == 'Minyak Goreng':
        model_minyak = load_model('model_prediksi_minyak_goreng_lstm_lag30.h5')
        model = model_minyak
        kode = 'group_keringan/Q22'
        lag = 30
    elif komoditas == 'Daging Ayam':
        model_ayam = load_model('model_prediksi_daging_ayam_lstm_lag30.h5')
        model = model_ayam
        kode = 'group_ayam/Q7'
        lag = 30
    elif komoditas == 'Daging Sapi':
        model_sapi = load_model('model_prediksi_daging_sapi_lstm_lag30.h5')
        model = model_sapi
        kode = 'group_daging/Q6'
        lag = 30
    elif komoditas == 'Telur Ayam':
        model_telur = load_model('model_prediksi_telur_ayam_lstm_lag30.h5')
        model = model_telur
        kode = 'group_keringan/Q25'
        lag = 30
    elif komoditas == 'Bawang Merah':
        model_bamer = load_model('model_prediksi_bawang_merah_lstm_lag30.h5')
        model = model_bamer
        kode = 'group_sayur/Q9'
        lag = 30
    elif komoditas == 'Bawang Putih':
        model_baput = load_model('model_prediksi_bawang_putih_lstm_lag30.h5')
        model = model_baput
        kode = 'group_sayur/Q10'
        lag = 30
    elif komoditas == 'Gula Pasir':
        model_gula = load_model('model_prediksi_gula_pasir_lstm.h5')
        model = model_gula
        kode = 'group_sayur/Q17'
        lag = 60
    else:
        print('Pilih Komoditas')
    
    return model, lag, kode
    
def main():
    
    # LOAD MODEL
    lag = 60
    st.title("Prediksi Harga Bahan Pokok")
    komoditas = st.selectbox(
        "Jenis Komoditas",
        ("Beras Premium", "Bawang Merah", "Bawang Putih","Cabai Merah", "Daging Ayam","Daging Sapi","Gula Pasir","Minyak Goreng", "Telur Ayam"),
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
    )
    
    model, lag, kode = load_model_ml(komoditas)
    
    # ------------------- PROSES data
    resp_dict = get_data_api()

    list_data = []
    for i in range(len(resp_dict)):
        list_data.append((resp_dict[i].get('Q1'),resp_dict[i].get(kode)))

    awal = (datetime.now() - relativedelta(months=5)).strftime('%Y-%m-%d')
    akhir  = (datetime.now() - relativedelta(days=1)).strftime('%Y-%m-%d')
    df = pd.DataFrame(list_data, columns = ["Tanggal",'Harga'])
    df = df[(df['Tanggal']>=awal) & (df['Tanggal']<=akhir)]

    df['Tanggal'] = pd.to_datetime(df['Tanggal'],  format='%Y-%m-%d') 
    #df = df.set_index('Tanggal').resample('1D').median()
    df = df.groupby(['Tanggal']).median()
    idx = pd.date_range(df.index.min(),df.index.max())

    df.index = pd.DatetimeIndex(df.index)

    s = df.reindex(idx, method='nearest')

    data = s.reset_index()
    data.columns = ["Tanggal", "Harga"]
    df = data[['Tanggal','Harga']].dropna()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'],  format='%Y-%m-%d') 
    data['Harga']=data['Harga'].astype(int)
    sc = MinMaxScaler(feature_range=(0,1))

    hari = st.slider("Tentukan jumlah hari", 1, 30,step=1)
    

    if st.button('Prediksi'):
        data_baru = data.copy()
        X_test = []
        pred = []
        th = np.arange(5000,300000,500)
        for i in range(hari):
            test_set_scaled = sc.fit_transform(data_baru.iloc[len(data_baru)-lag:,1:2].values)
            X_test = [test_set_scaled[:, 0]]
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            tgl = (data_baru['Tanggal'].iloc[-1] + relativedelta(days=1)).strftime('%Y-%m-%d')
            hasil_pred = predict(X_test, model)
            hasil_pred = sc.inverse_transform(hasil_pred)
            selisih = abs(hasil_pred[0][0]-th)
            print(selisih)
            print(th)
            hasil_pred = th[list(selisih).index(min(selisih))]
            pred.append((tgl,hasil_pred))

            #update data_baru
            data_baru.loc[len(data_baru)] = [tgl,hasil_pred]
            data_baru['Tanggal'] = pd.to_datetime(data_baru['Tanggal'],  format='%Y-%m-%d')
        
        data_pred = pd.DataFrame(pred, columns = ["Tanggal", "Harga"])
        print("------------Data Prediksi------------")
        print(data_pred)
        print("-----------Data eksisting + Prediksi --------")
        print(data_baru)

        col1,col2 = st.columns([2,3])
        with col1:
            #data_pred['Harga'] = data_pred['Harga'].reshape(-1,1)
            st.dataframe(data_pred)
        with col2:
            fig, ax = plt.subplots()
            #data_baru.set_index('Tanggal')
            
            #ax = data_baru.iloc[:len(data_baru)-hari,1].plot(ls="-", color="gray", label='Harga saat ini')
            #data_baru.iloc[len(data_baru)-hari:,1].plot(ls="-", color="blue", label='Harga Prediksi', ax=ax)
            
            ax.plot(data_baru['Tanggal'].iloc[:len(data_baru)-hari], data_baru['Harga'].iloc[:len(data_baru)-hari], label='Harga Saat ini')
            ax.plot(data_baru['Tanggal'].iloc[len(data_baru)-hari:], data_baru['Harga'].iloc[len(data_baru)-hari:], label = 'Harga Prediksi')
            ax.set(xlabel='Tanggal', ylabel='Harga (Rp)')
            plt.legend()
            fig.set_figwidth(6) 
            fig.set_figheight(4)
            st.plotly_chart(fig, use_container_width=False)

    

if __name__ == "__main__":
    main()