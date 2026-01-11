"""
Module tải dữ liệu từ CSV và gộp thành một tập dữ liệu hoàn chỉnh.
Sử dụng Singleton pattern và caching Streamlit để tối ưu hiệu năng.
"""
import os
import pandas as pd
import streamlit as st
from typing import Optional

class DataLayerSingleton:
    """
    Singleton class để quản lý tải dữ liệu từ CSV.
    - Lấy mẫu dữ liệu để tối ưu bộ nhớ.
    - Cache kết quả lên bộ nhớ của Streamlit.
    """
    _instance = None
    
    def __new__(cls):
        """Đảm bảo chỉ có một instance duy nhất."""
        if cls._instance is None:
            cls._instance = super(DataLayerSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Khởi tạo Singleton (chỉ chạy một lần)."""
        if self._initialized:
            return
        self._initialized = True
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        
    @st.cache_data
    def load_transaction_data(_self, sample_size: Optional[int] = 50000) -> pd.DataFrame:
        """
        Tải dữ liệu giao dịch từ CSV.
        
        Tham số:
        - sample_size (int): Số lượng dòng để lấy mẫu. Nếu None, tải toàn bộ.
                            Mặc định 50000 để tối ưu hiệu năng.
        
        Trả về:
        - pd.DataFrame: Dữ liệu giao dịch (cột: household_key, BASKET_ID, DAY, 
                        PRODUCT_ID, QUANTITY, SALES_VALUE, TRANS_TIME, WEEK_NO)
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'transaction_data.csv'))
            if sample_size is not None and len(df) > sample_size:
                df = df.head(sample_size)
            return df
        except Exception as e:
            st.error(f"Lỗi tải transaction_data.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_product_data(_self) -> pd.DataFrame:
        """
        Tải dữ liệu sản phẩm từ CSV.
        
        Trả về:
        - pd.DataFrame: Dữ liệu sản phẩm (cột: PRODUCT_ID, MANUFACTURER, 
                        DEPARTMENT, COMMODITY_DESC)
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'product.csv'))
            return df
        except Exception as e:
            st.error(f"Lỗi tải product.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_demographic_data(_self) -> pd.DataFrame:
        """
        Tải dữ liệu nhân khẩu học của hộ gia đình.
        
        Trả về:
        - pd.DataFrame: Dữ liệu nhân khẩu học (cột: household_key, AGE_DESC, 
                        MARITAL_STATUS_CODE, INCOME_DESC, HOMEOWNER_DESC)
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'hh_demographic.csv'))
            return df
        except Exception as e:
            st.error(f"Lỗi tải hh_demographic.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_campaign_data(_self) -> pd.DataFrame:
        """
        Tải dữ liệu chiến dịch (nhãn mục tiêu).
        
        Trả về:
        - pd.DataFrame: Dữ liệu chiến dịch (cột: household_key, CAMPAIGN)
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'campaign_table.csv'))
            return df
        except Exception as e:
            st.error(f"Lỗi tải campaign_table.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_coupon_data(_self) -> pd.DataFrame:
        """
        Tải dữ liệu phiếu giảm giá.
        
        Trả về:
        - pd.DataFrame: Dữ liệu phiếu giảm giá
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'coupon.csv'))
            return df
        except Exception as e:
            st.error(f"Lỗi tải coupon.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_coupon_redemption_data(_self) -> pd.DataFrame:
        """
        Tải dữ liệu hoàn đổi phiếu giảm giá.
        
        Trả về:
        - pd.DataFrame: Dữ liệu hoàn đổi phiếu giảm giá
        """
        try:
            df = pd.read_csv(os.path.join(_self.data_path, 'coupon_redempt.csv'))
            return df
        except Exception as e:
            st.error(f"Lỗi tải coupon_redempt.csv: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def get_merged_dataset(_self, sample_size: Optional[int] = 50000) -> pd.DataFrame:
        """
        Gộp dữ liệu giao dịch với thông tin sản phẩm và nhân khẩu học.
        
        Tham số:
        - sample_size (int): Số lượng dòng để lấy mẫu từ transaction_data.
        
        Trả về:
        - pd.DataFrame: Dữ liệu gộp
        """
        try:
            trans = _self.load_transaction_data(sample_size)
            product = _self.load_product_data()
            demo = _self.load_demographic_data()
            
            # Gộp giao dịch với sản phẩm
            merged = trans.merge(product, on='PRODUCT_ID', how='left')
            
            # Gộp với nhân khẩu học
            merged = merged.merge(demo, on='household_key', how='left')
            
            return merged
        except Exception as e:
            st.error(f"Lỗi khi gộp dữ liệu: {str(e)}")
            return pd.DataFrame()


def get_data_layer() -> DataLayerSingleton:
    """
    Hàm tiện ích để lấy instance duy nhất của DataLayerSingleton.
    
    Trả về:
    - DataLayerSingleton: Instance duy nhất
    """
    return DataLayerSingleton()
