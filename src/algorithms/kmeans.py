"""
Thuật toán k-Means Clustering - Phân nhóm không giám sát (Unsupervised Learning).
Phân nhóm khách hàng dựa trên chỉ số RFM (Recency, Frequency, Monetary).
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple
from datetime import datetime, timedelta
from src.config import KMEANS_CONFIG

class KMeansClustering:
    """
    Lớp triển khai k-Means Clustering.
    
    k-Means là thuật toán phân nhóm không giám sát (Unsupervised) chia dữ liệu
    thành k cụm (clusters) sao cho các điểm trong cùng cụm gần nhau
    và các cụm khác xa nhau.
    
    RFM (Recency, Frequency, Monetary):
    - Recency: Thời gian kể từ lần mua cuối cùng
    - Frequency: Số lần mua trong khoảng thời gian
    - Monetary: Tổng chi tiêu
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Khởi tạo k-Means Clustering.
        
        Tham số:
        - n_clusters (int): Số lượng cụm
        - random_state (int): Seed cho tái tạo kết quả
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, 
                            n_init=10)
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.cluster_labels = None
        self.rfm_data = None
    
    def calculate_rfm(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính chỉ số RFM từ dữ liệu giao dịch.
        
        Tham số:
        - transaction_df (pd.DataFrame): Dữ liệu giao dịch
        
        Trả về:
        - pd.DataFrame: Dữ liệu RFM theo household_key
        """
        try:
            # DAY là số ngày (int), không phải chuỗi ngày tháng
            # Lấy ngày tối đa (số ngày lớn nhất)
            max_day = transaction_df['DAY'].max()
            
            rfm = transaction_df.groupby('household_key').agg({
                'DAY': lambda x: max_day - x.max(),  # Recency: số ngày kể từ giao dịch cuối
                'BASKET_ID': 'nunique',              # Frequency: số lần mua khác nhau
                'SALES_VALUE': 'sum'                 # Monetary: tổng chi tiêu
            }).reset_index()
            
            rfm.columns = ['household_key', 'Recency', 'Frequency', 'Monetary']
            
            # Xử lý giá trị NaN (nếu có)
            rfm = rfm.dropna()
            rfm = rfm[rfm['Monetary'] > 0]  # Loại bỏ khách hàng không chi tiêu
            
            # Chuyển Recency thành int nếu cần
            rfm['Recency'] = rfm['Recency'].astype(int)
            rfm['Frequency'] = rfm['Frequency'].astype(int)
            
            self.rfm_data = rfm
            return rfm
        
        except Exception as e:
            st.error(f"Lỗi tính RFM: {str(e)}")
            return pd.DataFrame()
    
    def fit(self, transaction_df: pd.DataFrame) -> Dict:
        """
        Huấn luyện mô hình k-Means.
        
        Tham số:
        - transaction_df (pd.DataFrame): Dữ liệu giao dịch
        
        Trả về:
        - Dict: Kết quả huấn luyện
        """
        try:
            # Tính RFM
            rfm = self.calculate_rfm(transaction_df)
            
            if len(rfm) == 0:
                st.warning("Không thể tính RFM từ dữ liệu.")
                return {'error': 'Dữ liệu RFM rỗng'}
            
            # Chọn các cột RFM
            X = rfm[['Recency', 'Frequency', 'Monetary']].values
            
            # Chuẩn hóa dữ liệu (standardization)
            self.X_scaled = self.scaler.fit_transform(X)
            
            # Huấn luyện mô hình
            self.model.fit(self.X_scaled)
            self.cluster_labels = self.model.labels_
            
            # Thêm nhãn cụm vào dữ liệu RFM
            rfm['Cluster'] = self.cluster_labels
            
            # Tính các chỉ số đánh giá
            silhouette = silhouette_score(self.X_scaled, self.cluster_labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, self.cluster_labels)
            inertia = self.model.inertia_
            
            return {
                'rfm_data': rfm,
                'cluster_centers': self.model.cluster_centers_,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'inertia': inertia,
                'n_clusters': self.n_clusters,
                'n_samples': len(rfm)
            }
        
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện k-Means: {str(e)}")
            return {'error': str(e)}
    
    def get_cluster_statistics(self, rfm_data: pd.DataFrame) -> Dict:
        """
        Tính thống kê cho mỗi cụm.
        
        Tham số:
        - rfm_data (pd.DataFrame): Dữ liệu RFM với nhãn cụm
        
        Trả về:
        - Dict: Thống kê cho mỗi cụm
        """
        stats = {}
        for cluster_id in range(self.n_clusters):
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster_id]
            stats[f'Cụm {cluster_id}'] = {
                'Số khách hàng': len(cluster_data),
                'Recency trung bình': cluster_data['Recency'].mean(),
                'Frequency trung bình': cluster_data['Frequency'].mean(),
                'Monetary trung bình': cluster_data['Monetary'].mean()
            }
        
        return stats
    
    def predict(self, new_rfm: np.ndarray) -> np.ndarray:
        """
        Dự đoán cụm cho dữ liệu mới.
        
        Tham số:
        - new_rfm (np.ndarray): Dữ liệu RFM mới (shape: [n, 3])
        
        Trả về:
        - np.ndarray: Nhãn cụm dự đoán
        """
        if self.model is None:
            st.error("Mô hình chưa được huấn luyện.")
            return None
        
        new_rfm_scaled = self.scaler.transform(new_rfm)
        return self.model.predict(new_rfm_scaled)
