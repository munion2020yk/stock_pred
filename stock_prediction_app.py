import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import FinanceDataReader as fdr
import yfinance as yf # yfinance 추가
from pykrx.stock import get_market_fundamental
from datetime import datetime, timedelta
import warnings
import time
import sys # sys 추가

# --- 0. 설정 및 경고 무시 ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="주가 예측 앱")

# --- 1. 모델 및 스케일러 클래스 정의 (훈련 스크립트와 동일) ---
# (Streamlit 앱은 이 클래스 정의를 알아야 .pkl과 .pth 파일을 로드할 수 있습니다)

class ManualMinMaxScaler:
    """
    sklearn.preprocessing.MinMaxScaler를 대체하기 위한 수동 구현 클래스.
    """
    def __init__(self):
        self.data_min_ = None
        self.scale_ = None
    def fit(self, data):
        self.data_min_ = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        self.scale_ = data_max - self.data_min_ + 1e-8
    def transform(self, data):
        if self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted. Call 'fit' first.")
        return (data - self.data_min_) / self.scale_
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    def inverse_transform(self, data):
        if self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted. Call 'fit' first.")
        return data * self.scale_ + self.data_min_

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, kernel_size, hidden_size, num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, cnn_out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x_cnn = x.permute(0, 2, 1) # (batch_size, input_size, sequence_length)
        x_cnn = self.relu(self.conv1d(x_cnn)) # (batch_size, cnn_out_channels, sequence_length)
        x_lstm = x_cnn.permute(0, 2, 1) # (batch_size, sequence_length, cnn_out_channels)
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # We only want the last time step's output
        prediction = self.fc(lstm_out[:, -1, :]) # (batch_size, output_size)
        return prediction

class CNNOnlyModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, kernel_size, sequence_length, output_size):
        super(CNNOnlyModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, cnn_out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Calculate the input size for the fully connected layer
        fc_input_size = cnn_out_channels * sequence_length
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x_cnn = x.permute(0, 2, 1) # (batch_size, input_size, sequence_length)
        x_cnn = self.relu(self.conv1d(x_cnn)) # (batch_size, cnn_out_channels, sequence_length)
        # Flatten the output of Conv1d
        x_flat = self.flatten(x_cnn.permute(0, 2, 1)) # (batch_size, sequence_length * cnn_out_channels)
        # Note: Permute before flatten might be needed if flatten operation is not dim-specific
        # Let's test flatten(x_cnn) directly
        x_flat = self.flatten(x_cnn) # (batch_size, cnn_out_channels * sequence_length)
        prediction = self.fc(x_flat) # (batch_size, output_size)
        return prediction

# --- 2. 데이터 수집 함수 (실시간) ---
# @st.cache_data: Streamlit 캐시를 사용하여 동일한 종목은 빠르게 다시 로드
@st.cache_data(ttl=3600) # 1시간 동안 캐시
def get_prediction_data(ticker, sequence_length, feature_columns_list):
    """
    예측에 필요한 최신 데이터를 수집하고 전처리합니다.
    (samsung_data_collector.py의 축소/일반화 버전)
    """
    
    # 넉넉하게 90일치 데이터 수집 (주말, 휴일 포함)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    start_pykrx = start_date.strftime('%Y%m%d')
    end_pykrx = end_date.strftime('%Y%m%d')

    data_dfs = {}
    base_index = None # 모든 데이터를 정렬할 기준 인덱스

    # 1. 개별 종목 데이터 (OHLCV) - fdr
    try:
        stock_df = fdr.DataReader(ticker, start_str, end_str)
        if stock_df.empty:
            raise ValueError(f"fdr.DataReader({ticker}) returned no data.")
        stock_df = stock_df.rename(columns={
            'Open': 'Stock_Open', 'High': 'Stock_High', 'Low': 'Stock_Low',
            'Close': 'Stock_Close', 'Volume': 'Stock_Volume'
        })
        data_dfs['stock'] = stock_df[['Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 'Stock_Volume']]
        base_index = stock_df.index # 개별 종목의 개장일 기준
    except Exception as e:
        st.error(f"종목({ticker}) OHLCV 데이터 수집 실패: {e}")
        return None, None

    # 2. 개별 종목 데이터 (PER, PBR) - pykrx
    try:
        if 'Stock_PER' in feature_columns_list or 'Stock_PBR' in feature_columns_list:
            time.sleep(0.5) # API Rate Limit
            fund_df = get_market_fundamental(start_pykrx, end_pykrx, ticker)
            fund_df = fund_df[['PER', 'PBR']]
            fund_df = fund_df.rename(columns={'PER': 'Stock_PER', 'PBR': 'Stock_PBR'})
            fund_df.index = pd.to_datetime(fund_df.index)
            data_dfs['fund'] = fund_df
        else:
            # 훈련에 PER/PBR이 없으면 수집 X
            data_dfs['fund'] = pd.DataFrame(index=base_index, columns=['Stock_PER', 'Stock_PBR'])
            
    except Exception as e:
        st.warning(f"종목({ticker}) PER/PBR 데이터 수집 실패 (ffill로 채워짐): {e}")
        # 실패 시 빈 데이터프레임 (나중에 join 후 ffill)
        data_dfs['fund'] = pd.DataFrame(index=base_index, columns=['Stock_PER', 'Stock_PBR'])

    # 3. 매크로 데이터
    macro_tickers = {
        'KOSPI_Close': 'KS11',      # fdr
        'NAS_Close': 'IXIC',        # fdr
        'WTI_Close': 'WTI',         # fdr
        'USD_KRW_Close': 'USD/KRW', # fdr
        'VKOSPI_close': '^VKOSPI'   # yfinance
    }
    
    # 훈련에 사용된 피처만 수집
    active_macro_tickers = {}
    for col, tick in macro_tickers.items():
        if col in feature_columns_list:
            active_macro_tickers[col] = tick
            
    for col_name, ticker_symbol in active_macro_tickers.items():
        try:
            time.sleep(0.2)
            if ticker_symbol.startswith('^'): # yfinance
                df = yf.Ticker(ticker_symbol).history(start=start_str, end=end_str)[['Close']]
            else: # fdr
                df = fdr.DataReader(ticker_symbol, start_str, end_str)[['Close']]
            
            if df.empty:
                st.warning(f"매크로 데이터({col_name}, {ticker_symbol}) 수집 결과가 비어있습니다. 계속 진행합니다.")
                df = pd.DataFrame(columns=[col_name])
            else:
                df.columns = [col_name]
                # yfinance는 UTC 기준 시간을 반환할 수 있으므로, 날짜만 사용
                df.index = pd.to_datetime(df.index.date)
            
            data_dfs[col_name] = df
            
        except Exception as e:
            st.error(f"매크로 데이터({col_name}, {ticker_symbol}) 수집 실패: {e}")
            return None, None
            
    # 4. 데이터 병합 및 후처리
    try:
        base_df = data_dfs['stock']
        for key, df in data_dfs.items():
            if key != 'stock':
                # base_df(개장일) 기준으로 매크로 데이터 병합
                base_df = base_df.join(df, how='left')
        
        # 훈련 스크립트와 동일한 전처리
        # 1. ffill (휴일 등에 매크로 데이터 채우기)
        base_df = base_df.ffill()
        
        # 2. 시차 적용
        if 'NAS_Close' in base_df.columns:
            base_df['NAS_Close'] = base_df['NAS_Close'].shift(1)
        if 'WTI_Close' in base_df.columns:
            base_df['WTI_Close'] = base_df['WTI_Close'].shift(1)
            
        # 3. bfill (시차 적용으로 생긴 맨 앞 NaN 채우기)
        base_df = base_df.bfill() 
        
        # 4. PER/PBR이 0인 경우 (수집 실패 등)
        if 'Stock_PER' in base_df.columns:
            base_df['Stock_PER'] = base_df['Stock_PER'].fillna(1e-4).replace(0, 1e-4)
        if 'Stock_PBR' in base_df.columns:
            base_df['Stock_PBR'] = base_df['Stock_PBR'].fillna(1e-4).replace(0, 1e-4)

        # 5. feature_columns_list에 있는 컬럼만 선택
        final_df = base_df[feature_columns_list]

        # 6. 최종 NaN 확인
        if final_df.isnull().values.any():
            st.warning("데이터 수집 후에도 NaN 값이 남아있습니다. 0으로 채웁니다.")
            st.dataframe(final_df[final_df.isnull().any(axis=1)])
            final_df = final_df.fillna(0)

        if len(final_df) < sequence_length:
            st.error(f"데이터 수집 후 {sequence_length}일치의 데이터가 남지 않았습니다. (수집된 데이터 {len(final_df)}일치)")
            return None, None

        # 마지막 종가 (등락률 복원 시 기준점)
        last_price = final_df['Stock_Close'].iloc[-1]
        
        # 최종 입력 데이터 (마지막 30일)
        final_input_data = final_df.iloc[-sequence_length:]
        
        return final_input_data, last_price

    except Exception as e:
        st.error(f"데이터 병합 및 후처리 실패: {e}")
        st.exception(e) # 자세한 오류 로그
        return None, None

# --- 3. Artifacts 로드 함수 ---
# @st.cache_resource: 모델/스케일러처럼 큰 객체는 리소스 캐시에 저장
@st.cache_resource
def load_artifacts():
    """훈련된 모델과 스케일러를 로드합니다."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        with open('feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"필수 파일(.pkl)을 찾을 수 없습니다: {e}")
        st.error("먼저 'model_trainer.py'를 실행하여 Artifacts를 생성해야 합니다.")
        st.stop() # 앱 실행 중지
        
    try:
        # 모델 파라미터 정의 (훈련 스크립트와 동일)
        input_size = len(feature_columns)
        output_size = 5
        sequence_length = 30
        cnn_out = 64
        kernel = 3
        lstm_hidden = 128
        lstm_layers = 2
        
        model_lstm = CNNLSTMModel(input_size, cnn_out, kernel, lstm_hidden, lstm_layers, output_size).to(device)
        model_lstm.load_state_dict(torch.load('best_cnn_lstm_model.pth', map_location=device))
        model_lstm.eval()
        
        model_cnn = CNNOnlyModel(input_size, cnn_out, kernel, sequence_length, output_size).to(device)
        model_cnn.load_state_dict(torch.load('best_cnn_only_model.pth', map_location=device))
        model_cnn.eval()
        
    except FileNotFoundError as e:
        st.error(f"필수 모델 파일(.pth)을 찾을 수 없습니다: {e}")
        st.error("먼저 'model_trainer.py'를 실행하여 Artifacts를 생성해야 합니다.")
        st.stop()
    except Exception as e:
        st.error(f"모델 로드 중 알 수 없는 오류 발생: {e}")
        st.exception(e)
        st.stop()

    return model_lstm, model_cnn, feature_scaler, target_scaler, feature_columns, device

# --- 4. Streamlit UI 메인 함수 ---
def main():
    st.title("📈 딥러닝 주가 예측 웹 앱 (T+1 ~ T+5)")
    st.markdown("훈련된 **CNN+LSTM** 및 **CNN-Only** 모델을 사용하여 선택한 종목의 5영업일 후 종가를 예측합니다.")

    # 1. Artifacts 로드
    artifacts = load_artifacts()
    if artifacts is None:
        return
        
    model_lstm, model_cnn, feature_scaler, target_scaler, feature_columns, device = artifacts
    
    # 2. 사용자 입력
    st.sidebar.header("예측 옵션")
    ticker = st.sidebar.text_input("종목 코드를 입력하세요 (예: 000660)", "005930")
    
    if st.sidebar.button("예측 실행"):
        if not ticker.isdigit() or len(ticker) != 6:
            st.sidebar.error("올바른 6자리 종목 코드를 입력하세요.")
            return

        # 3. 데이터 수집
        with st.spinner(f"'{ticker}' 종목의 최신 {len(feature_columns)}개 피처 데이터를 수집 및 전처리 중..."):
            input_data_df, last_price = get_prediction_data(ticker, sequence_length=30, feature_columns_list=feature_columns)
        
        if input_data_df is None:
            # get_prediction_data 함수 내부에서 이미 에러 메시지 표시됨
            return
            
        st.success(f"'{ticker}' 데이터 수집 완료. (기준 종가: {last_price:,.0f} 원)")

        # 4. 예측 수행
        with st.spinner("모델 예측 수행 중..."):
            try:
                # 4-1. 훈련 시 사용된 컬럼 순서대로 정렬 (get_prediction_data에서 이미 수행됨)
                input_np_unscaled = input_data_df.values
                num_features = len(feature_columns)
                
                # 4-2. 스케일링
                input_np_scaled = feature_scaler.transform(input_np_unscaled.reshape(-1, num_features)).reshape(input_np_unscaled.shape)
                
                # 4-3. 텐서 변환 (배치 크기 1)
                input_tensor = torch.tensor(input_np_scaled, dtype=torch.float32).unsqueeze(0).to(device) # (1, 30, num_features)
                
                # 4-4. 모델 예측
                with torch.no_grad():
                    pred_lstm_scaled = model_lstm(input_tensor)
                    pred_cnn_scaled = model_cnn(input_tensor)
                    
                # 4-5. 스케일 복원 (등락률)
                pred_lstm_rates = target_scaler.inverse_transform(pred_lstm_scaled.cpu().numpy())[0] # (5,)
                pred_cnn_rates = target_scaler.inverse_transform(pred_cnn_scaled.cpu().numpy())[0] # (5,)
                
                # 4-6. 최종 가격 계산
                pred_lstm_prices = last_price * (1 + pred_lstm_rates)
                pred_cnn_prices = last_price * (1 + pred_cnn_rates)
                
            except Exception as e:
                st.error(f"예측 중 오류 발생: {e}")
                st.exception(e) # 자세한 오류 로그 표시
                return

        # 5. 결과 표시
        try:
            # KRX 종목 목록에서 이름 가져오기
            krx_listing = fdr.StockListing('KRX')
            stock_name = krx_listing.loc[ticker]['Name']
            st.subheader(f"종목: {stock_name} ({ticker}) (기준가: {last_price:,.0f}원)")
        except Exception:
            st.subheader(f"종목 코드: {ticker} (기준가: {last_price:,.0f}원)")
        
        col1, col2 = st.columns(2)
        
        # 결과 데이터프레임 생성
        days = [f"T+{i+1} (영업일)" for i in range(5)]
        results_df = pd.DataFrame({
            '시점': days,
            'CNN+LSTM 예측 종가 (원)': pred_lstm_prices,
            'CNN-Only 예측 종가 (원)': pred_cnn_prices
        })
        results_df['CNN+LSTM 예측 종가 (원)'] = results_df['CNN+LSTM 예측 종가 (원)'].apply(lambda x: f"{x:,.0f}")
        results_df['CNN-Only 예측 종가 (원)'] = results_df['CNN-Only 예측 종가 (원)'].apply(lambda x: f"{x:,.0f}")
        
        with col1:
            st.metric("T+1 예측 (CNN+LSTM)", f"{pred_lstm_prices[0]:,.0f} 원")
            st.metric("T+1 예측 (CNN-Only)", f"{pred_cnn_prices[0]:,.0f} 원")

        with col2:
            st.metric("T+5 예측 (CNN+LSTM)", f"{pred_lstm_prices[-1]:,.0f} 원")
            st.metric("T+5 예측 (CNN-Only)", f"{pred_cnn_prices[-1]:,.0f} 원")
            
        st.dataframe(results_df.set_index('시점'), use_container_width=True)

        with st.expander("최근 30일간의 입력 데이터 확인"):
            st.dataframe(input_data_df.tail(30), use_container_width=True)
            st.caption("참고: NAS_Close와 WTI_Close는 T-1일 기준 (시차 적용됨)이며, PER/PBR 수집 실패 시 0에 가까운 값으로 대체됩니다.")

if __name__ == "__main__":
    main()
