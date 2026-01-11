"""
Data Preprocessing & Analysis - Tiền xử lý và phân tích dữ liệu.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple

class DataPreprocessor:
    """Lớp xử lý tiền xử lý và phân tích dữ liệu."""
    
    def __init__(self):
        """Khởi tạo Data Preprocessor."""
        self.original_shape = None
        self.processed_shape = None
        self.statistics = {}
    
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        Phân tích dữ liệu chi tiết.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu cần phân tích
        
        Trả về:
        - Dict: Thống kê chi tiết
        """
        try:
            analysis = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'null_counts': df.isnull().sum().to_dict(),
                'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'column_info': {}
            }
            
            # Thông tin từng cột
            for col in df.columns:
                analysis['column_info'][col] = {
                    'dtype': str(df[col].dtype),
                    'unique': df[col].nunique(),
                    'null_count': df[col].isnull().sum(),
                    'null_pct': round(df[col].isnull().sum() / len(df) * 100, 2)
                }
            
            self.statistics = analysis
            return analysis
        
        except Exception as e:
            st.error(f"Lỗi khi phân tích dữ liệu: {str(e)}")
            return {}
    
    def preprocess_data(self, df: pd.DataFrame, 
                       remove_nulls: bool = True,
                       remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu gốc
        - remove_nulls (bool): Loại bỏ hàng có null
        - remove_duplicates (bool): Loại bỏ hàng trùng lặp
        
        Trả về:
        - pd.DataFrame: Dữ liệu đã xử lý
        """
        try:
            df_processed = df.copy()
            self.original_shape = df_processed.shape
            
            # Loại bỏ hàng trùng lặp
            if remove_duplicates:
                duplicates_before = len(df_processed)
                df_processed = df_processed.drop_duplicates()
                duplicates_removed = duplicates_before - len(df_processed)
            else:
                duplicates_removed = 0
            
            # Loại bỏ hàng có null values
            if remove_nulls:
                nulls_before = len(df_processed)
                df_processed = df_processed.dropna()
                nulls_removed = nulls_before - len(df_processed)
            else:
                nulls_removed = 0
            
            # Xóa các cột không cần thiết nếu toàn null
            all_null_cols = df_processed.columns[df_processed.isnull().all()].tolist()
            if all_null_cols:
                df_processed = df_processed.drop(columns=all_null_cols)
            
            self.processed_shape = df_processed.shape
            
            return df_processed, {
                'duplicates_removed': duplicates_removed,
                'nulls_removed': nulls_removed,
                'all_null_cols_removed': all_null_cols,
                'original_rows': self.original_shape[0],
                'processed_rows': self.processed_shape[0],
                'data_retention_pct': round(self.processed_shape[0] / self.original_shape[0] * 100, 2)
            }
        
        except Exception as e:
            st.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            return df, {}
    
    def get_statistics_summary(self) -> Dict:
        """Lấy tóm tắt thống kê."""
        return self.statistics
