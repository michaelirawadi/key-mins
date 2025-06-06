import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def preprocess(df: pd.DataFrame) -> np.ndarray:
    df_processed = df.copy()
    
    # Fill missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Label Encoder
    for col in df_processed.select_dtypes(include=['object', 'category']):
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Standard Scaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_processed)
    
    return data_scaled
