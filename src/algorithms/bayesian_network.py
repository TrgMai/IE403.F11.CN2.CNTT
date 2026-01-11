"""
Bayesian Network - Mô hình xác suất dựa trên Đồ thị có hướng (DAG).
Tìm mối quan hệ nhân quả giữa các biến: Tuổi -> Thu nhập -> Sở hữu nhà.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from src.config import BAYESIAN_NETWORK_CONFIG

class BayesianNetworkDAG:
    """
    Lớp triển khai Bayesian Network với DAG (Directed Acyclic Graph).
    
    Bayesian Network là một mô hình xác suất đồ thị biểu diễn các mối quan hệ
    phụ thuộc giữa các biến. Trong trường hợp này:
    - AGE -> INCOME: Tuổi ảnh hưởng đến thu nhập
    - INCOME -> HOMEOWNER: Thu nhập ảnh hưởng đến khả năng sở hữu nhà
    """
    
    def __init__(self):
        """Khởi tạo Bayesian Network."""
        self.data = None
        self.cpd_age = None
        self.cpd_income_given_age = None
        self.cpd_homeowner_given_income = None
        self.structure = {
            'nodes': ['AGE', 'INCOME', 'HOMEOWNER'],
            'edges': [('AGE', 'INCOME'), ('INCOME', 'HOMEOWNER')]
        }
    
    def encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mã hóa dữ liệu để sử dụng trong Bayesian Network.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu gốc
        
        Trả về:
        - pd.DataFrame: Dữ liệu đã mã hóa
        """
        df_encoded = df[['AGE_DESC', 'INCOME_DESC', 'HOMEOWNER_DESC']].copy()
        df_encoded = df_encoded.dropna()
        
        for col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def calculate_cpd_age(self, data: pd.DataFrame) -> Dict:
        """
        Tính Conditional Probability Distribution (CPD) cho biến AGE.
        
        CPD(AGE) = P(AGE) = số trường hợp AGE / tổng số trường hợp
        
        Tham số:
        - data (pd.DataFrame): Dữ liệu đã mã hóa
        
        Trả về:
        - Dict: CPD cho AGE
        """
        age_counts = data['AGE_DESC'].value_counts()
        cpd = age_counts / len(data)
        return cpd.to_dict()
    
    def calculate_cpd_income_given_age(self, data: pd.DataFrame) -> Dict:
        """
        Tính CPD cho biến INCOME có điều kiện AGE.
        
        CPD(INCOME|AGE) = P(INCOME|AGE) = số (INCOME, AGE) / số AGE
        
        Tham số:
        - data (pd.DataFrame): Dữ liệu đã mã hóa
        
        Trả về:
        - Dict: CPD cho INCOME|AGE
        """
        cpd = {}
        for age_val in data['AGE_DESC'].unique():
            age_subset = data[data['AGE_DESC'] == age_val]
            income_dist = age_subset['INCOME_DESC'].value_counts()
            cpd[age_val] = (income_dist / len(age_subset)).to_dict()
        
        return cpd
    
    def calculate_cpd_homeowner_given_income(self, data: pd.DataFrame) -> Dict:
        """
        Tính CPD cho biến HOMEOWNER có điều kiện INCOME.
        
        CPD(HOMEOWNER|INCOME) = P(HOMEOWNER|INCOME) = số (HOMEOWNER, INCOME) / số INCOME
        
        Tham số:
        - data (pd.DataFrame): Dữ liệu đã mã hóa
        
        Trả về:
        - Dict: CPD cho HOMEOWNER|INCOME
        """
        cpd = {}
        for income_val in data['INCOME_DESC'].unique():
            income_subset = data[data['INCOME_DESC'] == income_val]
            homeowner_dist = income_subset['HOMEOWNER_DESC'].value_counts()
            cpd[income_val] = (homeowner_dist / len(income_subset)).to_dict()
        
        return cpd
    
    def fit(self, df: pd.DataFrame) -> Dict:
        """
        Huấn luyện Bayesian Network.
        
        Tham số:
        - df (pd.DataFrame): Dữ liệu huấn luyện
        
        Trả về:
        - Dict: CPD của các biến
        """
        try:
            # Mã hóa dữ liệu
            self.data = self.encode_data(df)
            
            if len(self.data) == 0:
                st.warning("Dữ liệu sau mã hóa rỗng.")
                return {'error': 'Dữ liệu rỗng'}
            
            # Tính CPD
            self.cpd_age = self.calculate_cpd_age(self.data)
            self.cpd_income_given_age = self.calculate_cpd_income_given_age(self.data)
            self.cpd_homeowner_given_income = self.calculate_cpd_homeowner_given_income(self.data)
            
            return {
                'cpd_age': self.cpd_age,
                'cpd_income_given_age': self.cpd_income_given_age,
                'cpd_homeowner_given_income': self.cpd_homeowner_given_income,
                'structure': self.structure,
                'num_samples': len(self.data)
            }
        
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện Bayesian Network: {str(e)}")
            return {'error': str(e)}
    
    def predict_inference(self, age_value: int) -> Dict:
        """
        Suy diễn xác suất INCOME và HOMEOWNER dựa trên AGE.
        
        Tham số:
        - age_value (int): Giá trị AGE (đã mã hóa)
        
        Trả về:
        - Dict: Xác suất dự đoán cho INCOME và HOMEOWNER
        """
        try:
            if self.cpd_income_given_age is None:
                st.error("Mô hình chưa được huấn luyện.")
                return {}
            
            # P(INCOME|AGE)
            income_dist = self.cpd_income_given_age.get(age_value, {})
            
            # P(HOMEOWNER|INCOME) - tính trung bình trên tất cả INCOME
            homeowner_prob = {}
            for income_val in income_dist.keys():
                homeowner_given_income = self.cpd_homeowner_given_income.get(income_val, {})
                for homeowner_val, prob in homeowner_given_income.items():
                    if homeowner_val not in homeowner_prob:
                        homeowner_prob[homeowner_val] = 0
                    homeowner_prob[homeowner_val] += income_dist[income_val] * prob
            
            return {
                'P(INCOME|AGE)': income_dist,
                'P(HOMEOWNER|AGE)': homeowner_prob
            }
        
        except Exception as e:
            st.error(f"Lỗi khi suy diễn: {str(e)}")
            return {}
    
    def get_dag_structure(self) -> Dict:
        """
        Lấy cấu trúc DAG của Bayesian Network.
        
        Trả về:
        - Dict: Các nút (nodes) và cạnh (edges)
        """
        return self.structure
