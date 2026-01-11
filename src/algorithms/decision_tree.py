"""
Thuật toán Cây quyết định (Decision Tree):
- CART (Classification And Regression Trees): Sử dụng Gini Impurity
- C4.5 (Quinlan Algorithm): Sử dụng Information Gain
- ID3 (Iterative Dichotomiser 3): Sử dụng Entropy (biến thể của C4.5)
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple

class DecisionTreeCART:
    """
    Lớp triển khai Decision Tree CART.
    
    CART (Classification And Regression Trees) sử dụng Gini Impurity
    để chọn điểm phân chia tốt nhất tại mỗi nút.
    
    Gini Impurity = 1 - Σ(p_i)²
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        """
        Khởi tạo Decision Tree CART.
        
        Tham số:
        - max_depth (int): Độ sâu tối đa của cây
        - min_samples_split (int): Số mẫu tối thiểu để tách một nút
        """
        self.model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, 
                                             min_samples_split=min_samples_split)
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO',
                     test_size: float = 0.2) -> bool:
        """
        Chuẩn bị dữ liệu cho mô hình.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        - test_size (float): Tỷ lệ test set
        
        Trả về:
        - bool: Thành công hay không
        """
        try:
            df_clean = df.dropna()
            
            # Tạo cột mục tiêu nếu chưa có
            if target_column not in df_clean.columns:
                median = df_clean['SALES_VALUE'].median()
                df_clean[target_column] = (df_clean['SALES_VALUE'] > median).astype(int)
            
            # Chọn đặc trưng
            feature_candidates = ['AGE_DESC', 'INCOME_DESC', 'MARITAL_STATUS_CODE',
                                 'HOMEOWNER_DESC', 'QUANTITY']
            features = [col for col in feature_candidates if col in df_clean.columns]
            
            # Mã hóa
            df_encoded = df_clean[features + [target_column]].copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            X = df_encoded[features].values
            y = df_encoded[target_column].values
            
            self.feature_names = features
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            return True
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu: {str(e)}")
            return False
    
    def train(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO') -> Dict:
        """
        Huấn luyện mô hình.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        
        Trả về:
        - Dict: Kết quả huấn luyện
        """
        try:
            if not self.prepare_data(df, target_column):
                return {'error': 'Không thể chuẩn bị dữ liệu'}
            
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, self.y_pred)
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            # Feature importance
            feature_importance = {name: importance 
                                 for name, importance in 
                                 zip(self.feature_names, self.model.feature_importances_)}
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'num_nodes': self.model.tree_.node_count,
                'max_depth': self.model.get_depth(),
                'criterion': 'Gini Impurity'
            }
        except Exception as e:
            st.error(f"Lỗi huấn luyện CART: {str(e)}")
            return {'error': str(e)}


class DecisionTreeC45:
    """
    Lớp triển khai Decision Tree C4.5 (Quinlan Algorithm).
    
    C4.5 sử dụng Information Gain Ratio để chọn điểm phân chia.
    
    Information Gain = H(Parent) - Σ(|Child|/|Parent|) * H(Child)
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        """
        Khởi tạo Decision Tree C4.5.
        
        Tham số:
        - max_depth (int): Độ sâu tối đa của cây
        - min_samples_split (int): Số mẫu tối thiểu để tách một nút
        """
        # sklearn không có C4.5 riêng, nên sử dụng entropy (tương đương)
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth,
                                             min_samples_split=min_samples_split)
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO',
                     test_size: float = 0.2) -> bool:
        """
        Chuẩn bị dữ liệu cho mô hình.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        - test_size (float): Tỷ lệ test set
        
        Trả về:
        - bool: Thành công hay không
        """
        try:
            df_clean = df.dropna()
            
            if target_column not in df_clean.columns:
                median = df_clean['SALES_VALUE'].median()
                df_clean[target_column] = (df_clean['SALES_VALUE'] > median).astype(int)
            
            feature_candidates = ['AGE_DESC', 'INCOME_DESC', 'MARITAL_STATUS_CODE',
                                 'HOMEOWNER_DESC', 'QUANTITY']
            features = [col for col in feature_candidates if col in df_clean.columns]
            
            df_encoded = df_clean[features + [target_column]].copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            X = df_encoded[features].values
            y = df_encoded[target_column].values
            
            self.feature_names = features
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            return True
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu: {str(e)}")
            return False
    
    def train(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO') -> Dict:
        """
        Huấn luyện mô hình C4.5.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        
        Trả về:
        - Dict: Kết quả huấn luyện
        """
        try:
            if not self.prepare_data(df, target_column):
                return {'error': 'Không thể chuẩn bị dữ liệu'}
            
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, self.y_pred)
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            feature_importance = {name: importance 
                                 for name, importance in 
                                 zip(self.feature_names, self.model.feature_importances_)}
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'num_nodes': self.model.tree_.node_count,
                'max_depth': self.model.get_depth(),
                'criterion': 'Information Gain (Entropy)'
            }
        except Exception as e:
            st.error(f"Lỗi huấn luyện C4.5: {str(e)}")
            return {'error': str(e)}


class DecisionTreeID3:
    """
    Lớp triển khai Decision Tree ID3 (Iterative Dichotomiser 3).
    
    ID3 là phiên bản đơn giản của C4.5, sử dụng Information Gain (Entropy).
    Được triển khai bằng DecisionTreeClassifier với criterion='entropy'
    và max_depth=None (cây không giới hạn chiều sâu).
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        """
        Khởi tạo Decision Tree ID3.
        
        Tham số:
        - max_depth (int): Độ sâu tối đa (None = không giới hạn)
        - min_samples_split (int): Số mẫu tối thiểu để tách một nút
        """
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth,
                                             min_samples_split=min_samples_split)
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO',
                     test_size: float = 0.2) -> bool:
        """
        Chuẩn bị dữ liệu cho mô hình.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        - test_size (float): Tỷ lệ test set
        
        Trả về:
        - bool: Thành công hay không
        """
        try:
            df_clean = df.dropna()
            
            if target_column not in df_clean.columns:
                median = df_clean['SALES_VALUE'].median()
                df_clean[target_column] = (df_clean['SALES_VALUE'] > median).astype(int)
            
            feature_candidates = ['AGE_DESC', 'INCOME_DESC', 'MARITAL_STATUS_CODE',
                                 'HOMEOWNER_DESC', 'QUANTITY']
            features = [col for col in feature_candidates if col in df_clean.columns]
            
            df_encoded = df_clean[features + [target_column]].copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            X = df_encoded[features].values
            y = df_encoded[target_column].values
            
            self.feature_names = features
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            return True
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu: {str(e)}")
            return False
    
    def train(self, df: pd.DataFrame, target_column: str = 'CHI_TIEU_CAO') -> Dict:
        """
        Huấn luyện mô hình ID3.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target_column (str): Cột mục tiêu
        
        Trả về:
        - Dict: Kết quả huấn luyện
        """
        try:
            if not self.prepare_data(df, target_column):
                return {'error': 'Không thể chuẩn bị dữ liệu'}
            
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, self.y_pred)
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            feature_importance = {name: importance 
                                 for name, importance in 
                                 zip(self.feature_names, self.model.feature_importances_)}
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'num_nodes': self.model.tree_.node_count,
                'max_depth': self.model.get_depth(),
                'criterion': 'Entropy (Information Gain)'
            }
        except Exception as e:
            st.error(f"Lỗi huấn luyện ID3: {str(e)}")
            return {'error': str(e)}
