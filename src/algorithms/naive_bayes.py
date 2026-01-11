"""
Thuật toán Naïve Bayes - Phân lớp xác suất.
Bao gồm tùy chọn Laplace Smoothing để xử lý Zero Probability Problem.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
from src.config import NAIVE_BAYES_CONFIG

class NaiveBayesClassifier:
    """
    Lớp triển khai Naïve Bayes Classifier.
    
    Naïve Bayes giả định tất cả các đặc trưng độc lập với nhau (giả định Naïve).
    Được sử dụng để phân lớp khách hàng dựa trên mức chi tiêu.
    """
    
    def __init__(self, use_laplace_smoothing: bool = True):
        """
        Khởi tạo Naïve Bayes Classifier.
        
        Tham số:
        - use_laplace_smoothing (bool): Có sử dụng Laplace Smoothing không. Mặc định True.
        """
        self.model = None
        self.use_laplace_smoothing = use_laplace_smoothing
        self.feature_names = []
        self.encoders = {}
        self.target_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def prepare_classification_data(self, df: pd.DataFrame, 
                                     target_column: str = 'CHI_TIEU_CAO',
                                     test_size: float = 0.2,
                                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu phân lớp.
        Tạo cột mục tiêu dựa trên mức chi tiêu nếu chưa có.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu gốc
        - target_column (str): Tên cột mục tiêu
        - test_size (float): Tỷ lệ test set (0-1)
        - random_state (int): Seed cho tái tạo kết quả
        
        Trả về:
        - Tuple[np.ndarray, np.ndarray]: X (features) và y (target)
        """
        df_clean = df.dropna()
        
        # Tạo cột mục tiêu nếu chưa có (dựa trên SALES_VALUE)
        if target_column not in df_clean.columns:
            median_spending = df_clean['SALES_VALUE'].median()
            df_clean[target_column] = (df_clean['SALES_VALUE'] > median_spending).astype(int)
        
        # Chọn các đặc trưng số từ nhân khẩu học và giao dịch
        feature_candidates = ['AGE_DESC', 'INCOME_DESC', 'MARITAL_STATUS_CODE', 
                             'HOMEOWNER_DESC', 'QUANTITY', 'SALES_VALUE']
        
        features = [col for col in feature_candidates if col in df_clean.columns]
        
        if not features:
            st.warning("Không tìm thấy đặc trưng phù hợp.")
            return None, None
        
        # Mã hóa đặc trưng phân loại
        df_encoded = df_clean[features + [target_column]].copy()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.encoders[col] = le
        
        X = df_encoded[features].values
        y = df_encoded[target_column].values
        
        self.feature_names = features
        
        # Chia train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO') -> Dict:
        """
        Huấn luyện mô hình Naïve Bayes.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu huấn luyện
        - target_column (str): Tên cột mục tiêu
        
        Trả về:
        - Dict: Kết quả đánh giá mô hình (accuracy, precision, recall, confusion_matrix)
        """
        try:
            # Chuẩn bị dữ liệu
            X, y = self.prepare_classification_data(df, target_column)
            
            if X is None:
                return {'error': 'Không thể chuẩn bị dữ liệu'}
            
            # Tạo và huấn luyện mô hình
            if self.use_laplace_smoothing:
                # Gaussian NB với Laplace Smoothing (var_smoothing)
                self.model = GaussianNB(var_smoothing=1e-9)
            else:
                self.model = GaussianNB(var_smoothing=0)
            
            self.model.fit(self.X_train, self.y_train)
            
            # Dự đoán trên test set
            self.y_pred = self.model.predict(self.X_test)
            
            # Tính các chỉ số
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm,
                'num_train_samples': len(self.X_train),
                'num_test_samples': len(self.X_test),
                'feature_names': self.feature_names,
                'laplace_smoothing': self.use_laplace_smoothing
            }
        
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện Naïve Bayes: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán trên dữ liệu mới.
        
        Tham số:
        - X (np.ndarray): Dữ liệu đầu vào
        
        Trả về:
        - np.ndarray: Nhãn dự đoán
        """
        if self.model is None:
            st.error("Mô hình chưa được huấn luyện. Hãy chạy train trước.")
            return None
        
        return self.model.predict(X)
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Lấy xác suất dự đoán cho các lớp.
        
        Tham số:
        - X (np.ndarray): Dữ liệu đầu vào
        
        Trả về:
        - np.ndarray: Ma trận xác suất (n_samples, n_classes)
        """
        if self.model is None:
            st.error("Mô hình chưa được huấn luyện. Hãy chạy train trước.")
            return None
        
        return self.model.predict_proba(X)
