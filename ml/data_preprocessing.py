import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import numpy as np

# def preprocess(df: pd.DataFrame) -> np.ndarray:
#     df_processed = df.copy()
    
#     # Fill missing values
#     for col in df_processed.columns:
#         if df_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_processed[col]):
#             df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
#         else:
#             df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
#     # Label Encoder
#     for col in df_processed.select_dtypes(include=['object', 'category']):
#         le = LabelEncoder()
#         oh = OneHotEncoder(sparse_output=False)
#         # df_processed[col] = le.fit_transform(df_processed[col].astype(str))
#         df_processed[col] = oh.fit_transform(df_processed[col].astype(str))
    
#     # Standard Scaler
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(df_processed)
    
#     return data_scaled

# ==============================================================================================


def preprocess(df: pd.DataFrame) -> np.ndarray:
    df_processed = df.copy()

    # Fill missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    # Identify categorical and numerical columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df_processed.select_dtypes(exclude=['object', 'category']).columns

    # One-hot encode categorical columns if present
    if len(categorical_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False)
        encoded_array = ohe.fit_transform(df_processed[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array,
                                  columns=ohe.get_feature_names_out(categorical_cols),
                                  index=df_processed.index)
    else:
        encoded_df = pd.DataFrame(index=df_processed.index)

    # Keep numerical data if present
    if len(numerical_cols) > 0:
        numeric_df = df_processed[numerical_cols]
    else:
        numeric_df = pd.DataFrame(index=df_processed.index)

    # Combine encoded and numerical
    df_final = pd.concat([numeric_df, encoded_df], axis=1)

    if df_final.shape[1] == 0:
        raise ValueError("No features to process: both numerical and categorical columns are missing.")

    # Standard scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_final)

    return data_scaled