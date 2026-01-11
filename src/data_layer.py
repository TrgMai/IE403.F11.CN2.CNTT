"""
Data Layer - Truy cập và quản lý dữ liệu.
Theo Single Responsibility Principle: chỉ chịu trách nhiệm load và cache dữ liệu.

Hỗ trợ cả Local (từ data/) và Online (từ Kaggle)
"""
import os
import pandas as pd
import streamlit as st
from typing import Optional
from pathlib import Path
from src.config import DATA_FILES


class KaggleDownloader:
    """Lớp download dữ liệu từ Kaggle."""
    
    DATASET_ID = "vjchoudhary7/customer-segmentation-tutorial-in-python"
    
    @staticmethod
    def ensure_data_exists():
        """Download dữ liệu từ Kaggle nếu local không có."""
        try:
            # Kiểm tra nếu data/ folder đã có file
            data_path = Path("data")
            if data_path.exists() and len(list(data_path.glob("*.csv"))) > 0:
                return True  # Local data exists
            
            # Nếu không có local data, download từ Kaggle
            st.info("📥 Đang download dữ liệu từ Kaggle... Điều này có thể mất vài phút")
            
            import kagglehub
            
            # Download dataset
            path = kagglehub.dataset_download(KaggleDownloader.DATASET_ID)
            st.success("✅ Download thành công!")
            return True
            
        except ImportError:
            st.error("❌ kagglehub chưa cài đặt. Chạy: pip install kagglehub")
            return False
        except Exception as e:
            st.error(f"❌ Lỗi download từ Kaggle: {str(e)}\n\nHãy chắc chắn:\n1. Cài kagglehub: `pip install kagglehub`\n2. Setup Kaggle API credentials")
            return False


class DataLoader:
    """Lớp load dữ liệu từ file CSV."""
    
    @staticmethod
    def load_csv(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load dữ liệu từ file CSV."""
        try:
            # Nếu file không tồn tại local, cố gắng download từ Kaggle
            if not os.path.exists(file_path):
                st.warning(f"⚠️ Không tìm thấy {file_path}, đang cố download từ Kaggle...")
                if KaggleDownloader.ensure_data_exists():
                    # Thử lại
                    if os.path.exists(file_path):
                        return pd.read_csv(file_path, nrows=nrows)
                return pd.DataFrame()
            
            return pd.read_csv(file_path, nrows=nrows)
        except Exception as e:
            st.error(f"Lỗi load file {file_path}: {str(e)}")
            return pd.DataFrame()


class DataCache:
    """Lớp cache dữ liệu dùng Streamlit."""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cache_transaction_data(sample_size: Optional[int] = None) -> pd.DataFrame:
        """Cache dữ liệu giao dịch."""
        loader = DataLoader()
        return loader.load_csv(DATA_FILES['transaction'], nrows=sample_size)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cache_product_data() -> pd.DataFrame:
        """Cache dữ liệu sản phẩm."""
        loader = DataLoader()
        return loader.load_csv(DATA_FILES['product'])
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cache_demographic_data() -> pd.DataFrame:
        """Cache dữ liệu nhân khẩu học."""
        loader = DataLoader()
        return loader.load_csv(DATA_FILES['demographic'])
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cache_campaign_data() -> pd.DataFrame:
        """Cache dữ liệu chiến dịch."""
        loader = DataLoader()
        return loader.load_csv(DATA_FILES['campaign_table'])


class DataMerger:
    """Lớp merge dữ liệu từ nhiều nguồn."""
    
    @staticmethod
    def merge_all(trans_df: pd.DataFrame,
                  product_df: pd.DataFrame,
                  demo_df: pd.DataFrame) -> pd.DataFrame:
        """Merge tất cả dữ liệu."""
        try:
            merged = trans_df.merge(product_df, on='PRODUCT_ID', how='left')
            merged = merged.merge(demo_df, on='household_key', how='left')
            return merged
        except Exception as e:
            st.error(f"Lỗi merge dữ liệu: {str(e)}")
            return pd.DataFrame()


class DataLayerSingleton:
    """Singleton Pattern - Đảm bảo chỉ có 1 instance."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLayerSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.cache = DataCache()
        self.merger = DataMerger()
    
    def get_sample_size(self) -> Optional[int]:
        """
        Lấy sample_size từ session_state.
        Nếu mode = "full" → trả về None (load hết)
        Nếu mode = "custom" → trả về số lượng
        """
        if 'dataset_mode' not in st.session_state:
            return 30000  # Default
        
        if st.session_state.dataset_mode == "full":
            return None  # Load toàn bộ
        else:
            return st.session_state.sample_size  # Load custom amount
    
    def load_transaction_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load dữ liệu giao dịch."""
        if sample_size is None:
            sample_size = self.get_sample_size()
        return self.cache.cache_transaction_data(sample_size)
    
    def load_product_data(self) -> pd.DataFrame:
        """Load dữ liệu sản phẩm."""
        return self.cache.cache_product_data()
    
    def load_demographic_data(self) -> pd.DataFrame:
        """Load dữ liệu khách hàng."""
        return self.cache.cache_demographic_data()
    
    def load_campaign_data(self) -> pd.DataFrame:
        """Load dữ liệu chiến dịch."""
        return self.cache.cache_campaign_data()
    
    def get_merged_dataset(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load và merge tất cả dữ liệu."""
        if sample_size is None:
            sample_size = self.get_sample_size()
        
        trans = self.load_transaction_data(sample_size)
        product = self.load_product_data()
        demo = self.load_demographic_data()
        return self.merger.merge_all(trans, product, demo)


def get_data_layer() -> DataLayerSingleton:
    """Lấy singleton instance của DataLayer."""
    return DataLayerSingleton()

