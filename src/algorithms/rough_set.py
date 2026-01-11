"""
Thuật toán Rough Set - Lựa chọn đặc trưng (Feature Selection).
Sử dụng heuristic để chọn các đặc trưng quan trọng nhất.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Set, Tuple
from sklearn.preprocessing import LabelEncoder

class RoughSetReduct:
    """
    Lớp triển khai thuật toán Rough Set để lựa chọn đặc trưng.
    
    Rough Set sử dụng heuristic dựa trên thông tin để tìm tập hợp
    đặc trưng nhỏ nhất (Reduct) vẫn giữ được khả năng phân biệt dữ liệu.
    """
    
    def __init__(self):
        """Khởi tạo Rough Set Reduct."""
        self.reduct = []
        self.importance_scores = {}
    
    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa các đặc trưng phân loại thành số.
        
        Tham số:
        - df (pd.DataFrame): DataFrame chứa dữ liệu
        
        Trả về:
        - Tuple[pd.DataFrame, Dict]: DataFrame đã mã hóa và từ điển encoder
        """
        df_encoded = df.copy()
        encoders = {}
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        
        return df_encoded, encoders
    
    def calculate_information_gain(self, df: pd.DataFrame, 
                                   feature: str, 
                                   target: str) -> float:
        """
        Tính độ lợi thông tin (Information Gain) của một đặc trưng.
        
        Information Gain = H(Target) - H(Target|Feature)
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - feature (str): Tên đặc trưng
        - target (str): Tên cột mục tiêu
        
        Trả về:
        - float: Độ lợi thông tin
        """
        def entropy(series):
            """Tính Entropy của một chuỗi."""
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Entropy của target
        target_entropy = entropy(df[target])
        
        # Entropy có điều kiện theo feature
        conditional_entropy = 0
        for feature_value in df[feature].unique():
            subset = df[df[feature] == feature_value]
            subset_entropy = entropy(subset[target])
            weight = len(subset) / len(df)
            conditional_entropy += weight * subset_entropy
        
        # Information Gain
        gain = target_entropy - conditional_entropy
        return gain
    
    def greedy_reduct_selection(self, df: pd.DataFrame, 
                                 target: str,
                                 max_features: int = 5) -> List[str]:
        """
        Lựa chọn Reduct bằng phương pháp tham lam (Greedy) dựa trên Information Gain.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target (str): Tên cột mục tiêu
        - max_features (int): Số lượng đặc trưng tối đa để chọn
        
        Trả về:
        - List[str]: Danh sách các đặc trưng được chọn
        """
        features = [col for col in df.columns if col != target]
        reduct = []
        self.importance_scores = {}
        
        for _ in range(min(max_features, len(features))):
            best_feature = None
            best_gain = -1
            
            for feature in features:
                if feature not in reduct:
                    gain = self.calculate_information_gain(df, feature, target)
                    self.importance_scores[feature] = gain
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
            
            if best_feature is None or best_gain <= 0:
                break
            
            reduct.append(best_feature)
        
        self.reduct = reduct
        return reduct
    
    def run(self, df: pd.DataFrame, target: str, max_features: int = 5) -> Dict:
        """
        Chạy Rough Set Reduct.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu
        - target (str): Tên cột mục tiêu
        - max_features (int): Số lượng đặc trưng tối đa
        
        Trả về:
        - Dict: Kết quả chứa reduct và importance scores
        """
        try:
            # Mã hóa dữ liệu phân loại
            df_encoded, _ = self.encode_categorical_features(df)
            
            # Loại bỏ các hàng có giá trị NaN
            df_encoded = df_encoded.dropna()
            
            if len(df_encoded) == 0:
                st.warning("Dữ liệu sau khi xử lý rỗng.")
                return {'reduct': [], 'importance_scores': {}}
            
            # Lựa chọn Reduct
            reduct = self.greedy_reduct_selection(df_encoded, target, max_features)
            
            return {
                'reduct': reduct,
                'importance_scores': self.importance_scores,
                'num_features_selected': len(reduct)
            }
        
        except Exception as e:
            st.error(f"Lỗi khi chạy Rough Set Reduct: {str(e)}")
            return {'reduct': [], 'importance_scores': {}}
