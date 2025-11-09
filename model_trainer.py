import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import sys
import io

# --- 1. 전제조건: sklearn 금지 -> Manual Scaler 구현 ---
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

# --- 2. 데이터 준비 함수 (등락률 예측) ---
def create_sequences(features_unscaled, target_unscaled, sequence_length, prediction_horizon):
    """
    시계열 데이터를 LSTM에 입력할 시퀀스(window) 형태로 변환합니다.
    """
    X, y, base_prices = [], [], []
    
    if target_unscaled.ndim > 1:
        target_unscaled = target_unscaled.flatten()
        
    for i in range(len(features_unscaled) - sequence_length - prediction_horizon + 1):
        X.append(features_unscaled[i:i + sequence_length])
        
        last_price_in_window = target_unscaled[i + sequence_length - 1]
        
        if last_price_in_window == 0:
            temp_idx = i + sequence_length - 2
            while temp_idx >= 0 and target_unscaled[temp_idx] == 0:
                temp_idx -= 1
            last_price_in_window = target_unscaled[temp_idx] if temp_idx >= 0 else 1.0
        
        base_prices.append(last_price_in_window)

        target_prices = target_unscaled[i + sequence_length : i + sequence_length + prediction_horizon]
        y_rates = (target_prices / last_price_in_window) - 1
        y.append(y_rates)
        
    return np.array(X), np.array(y), np.array(base_prices).reshape(-1, 1)

# --- 3. 모델 정의 ---

# 3-1. CNN + LSTM 모델
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, kernel_size, hidden_size, num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, cnn_out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.relu(self.conv1d(x_cnn))
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

# 3-2. CNN-Only 모델
class CNNOnlyModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, kernel_size, sequence_length, output_size):
        super(CNNOnlyModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, cnn_out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        fc_input_size = cnn_out_channels * sequence_length
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.relu(self.conv1d(x_cnn))
        x_flat = self.flatten(x_cnn)
        prediction = self.fc(x_flat)
        return prediction

# --- 4. 훈련 및 저장 로직 ---
def train_and_save_model(model, model_name, train_loader, val_loader, epochs, lr, device):
    """범용 모델 훈련 함수"""
    print(f"\n--- Start Training ({model_name}) ---")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss_avg = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
             print(f"({model_name}) Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss_avg:.6f}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            
    print(f"--- Training Finished ({model_name}) ---")
    print(f"Best model saved to 'best_{model_name}_model.pth' (Val Loss: {best_val_loss:.6f})")

# --- 5. 메인 실행 로직 ---
def main():
    # --- 하이퍼파라미터 설정 ---
    FILE_PATH = 'samsung.csv'
    PREDICTION_HORIZON = 5
    SEQUENCE_LENGTH = 30
    
    CNN_OUT_CHANNELS = 64
    KERNEL_SIZE = 3
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    
    BATCH_SIZE = 32
    EPOCHS = 50 # (실제로는 더 길게 훈련하는 것이 좋습니다)
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. 데이터 로드 및 전처리 (오류 수정된 로직) ---
    data = None
    try:
        # 1. Try Tab + utf-8
        data = pd.read_csv(FILE_PATH, sep='\t', encoding='utf-8')
    except Exception as e1:
        print(f"Tab/utf-8 failed: {e1}. Trying Tab/utf-16...")
        try:
            # 2. Try Tab + utf-16
            with open(FILE_PATH, 'r', encoding='utf-16') as f:
                data = pd.read_csv(f, sep='\t')
        except Exception as e2:
            print(f"Tab/utf-16 failed: {e2}. Trying Comma/utf-8...")
            try:
                # 3. Try Comma + utf-8
                data = pd.read_csv(FILE_PATH, sep=',', encoding='utf-8')
            except Exception as e3:
                print(f"Comma/utf-8 failed: {e3}. Trying Comma/utf-16...")
                try:
                    # 4. Try Comma + utf-16
                    with open(FILE_PATH, 'r', encoding='utf-16') as f:
                        data = pd.read_csv(f, sep=',')
                except Exception as e4:
                    print(f"All loading attempts failed: {e4}")
                    sys.exit(1)

    if data is None:
        print("Error: Data could not be loaded.")
        sys.exit(1)
            
    print(f"Loaded data. Shape: {data.shape}")
    
    # [오류 방지] 열이 1개이거나 'Date' 열이 없는지 확인
    if data.shape[1] <= 1:
        print(f"Error: CSV loaded with only {data.shape[1]} column(s).")
        print("Please check the delimiter (tab or comma) of 'samsung.csv'.")
        sys.exit(1)
    if 'Date' not in data.columns:
        print(f"Error: 'Date' column not found. Available columns: {data.columns.tolist()}")
        sys.exit(1)

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').set_index('Date')
    
    # [중요] 컬럼명 일반화 (Generic Naming)
    # "Samsung_" 접두사를 "Stock_"으로 변경
    rename_cols = {col: col.replace('Samsung_', 'Stock_') for col in data.columns if col.startswith('Samsung_')}
    data = data.rename(columns=rename_cols)
    
    if data.isnull().values.any():
        print(f"NaN values found. Filling with 'ffill' and 'bfill'.")
        data = data.ffill()
        data = data.bfill()
    
    # [중요] 훈련에 사용할 컬럼 리스트 정의
    # (Streamlit 앱에서 이 순서대로 데이터를 가져와야 함)
    feature_columns = [
        'Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 'Stock_Volume',
        'KOSPI_Close', 'NAS_Close', 'WTI_Close', 'USD_KRW_Close', 
        'Stock_PER', 'Stock_PBR', 'VKOSPI_close'
    ]
    TARGET_COLUMN = 'Stock_Close'
    
    # 컬럼이 존재하는지 확인
    missing_cols = [col for col in feature_columns + [TARGET_COLUMN] if col not in data.columns]
    if missing_cols:
        print(f"Error: 다음 컬럼이 'samsung.csv'에 존재하지 않습니다: {missing_cols}")
        print(f"현재 컬럼: {data.columns.tolist()}")
        sys.exit(1)
        
    features_np = data[feature_columns].values
    target_np = data[TARGET_COLUMN].values

    # --- 2. 데이터 분할 ---
    n_total = len(data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_features_unscaled = features_np[:n_train]
    val_features_unscaled = features_np[n_train:n_train + n_val]
    # (Test data는 여기서는 사용하지 않고, 앱에서 실시간으로 사용)
    
    train_target_unscaled = target_np[:n_train]
    val_target_unscaled = target_np[n_train:n_train + n_val]

    # --- 3. 시퀀스 데이터 생성 ---
    X_train, y_train_rates, _ = create_sequences(train_features_unscaled, train_target_unscaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val_rates, _ = create_sequences(val_features_unscaled, val_target_unscaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    # --- 4. 데이터 스케일링 및 저장 ---
    num_features = X_train.shape[2]
    
    # 4-1. 피처 스케일러
    feature_scaler = ManualMinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    
    # 4-2. 타겟 스케일러
    target_scaler = ManualMinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_rates)
    y_val_scaled = target_scaler.transform(y_val_rates)
    
    print("\n--- Saving Artifacts ---")
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    print("Saved 'feature_scaler.pkl'")
    
    with open('target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    print("Saved 'target_scaler.pkl'")
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print("Saved 'feature_columns.pkl'")

    # --- 5. PyTorch DataLoader 생성 ---
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32), 
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32), 
        torch.tensor(y_val_scaled, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 6. 모델 훈련 ---
    
    # 6-1. CNN+LSTM 훈련
    model_lstm = CNNLSTMModel(
        input_size=num_features, 
        cnn_out_channels=CNN_OUT_CHANNELS, 
        kernel_size=KERNEL_SIZE, 
        hidden_size=LSTM_HIDDEN_SIZE, 
        num_layers=LSTM_NUM_LAYERS, 
        output_size=PREDICTION_HORIZON
    ).to(device)
    
    train_and_save_model(model_lstm, 'cnn_lstm', train_loader, val_loader, EPOCHS, LEARNING_RATE, device)

    # 6-2. CNN-Only 훈련
    model_cnn = CNNOnlyModel(
        input_size=num_features, 
        cnn_out_channels=CNN_OUT_CHANNELS, 
        kernel_size=KERNEL_SIZE, 
        sequence_length=SEQUENCE_LENGTH, 
        output_size=PREDICTION_HORIZON
    ).to(device)
    
    train_and_save_model(model_cnn, 'cnn_only', train_loader, val_loader, EPOCHS, LEARNING_RATE, device)

    print("\n--- All models trained and artifacts saved successfully. ---")

if __name__ == "__main__":
    main()