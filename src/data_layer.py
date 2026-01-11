"""
Data Layer - Truy cập và quản lý dữ liệu.
Theo Single Responsibility Principle: chỉ chịu trách nhiệm load và cache dữ liệu.

Hỗ trợ cả Local (từ data/) và Online (từ Kaggle)
"""
import os
import shutil
import pandas as pd
import streamlit as st
from typing import Optional
from pathlib import Path
from src.config import DATA_FILES


class KaggleDownloader:
    """Lớp download dữ liệu từ Kaggle."""
    
    DATASET_ID = "frtgnn/dunnhumby-the-complete-journey"
    
    @staticmethod
    def ensure_data_exists():
        """Download dữ liệu từ Kaggle nếu local không có."""
        try:
            data_path = Path("data")
            
            # 1. Kiểm tra nhanh: Nếu file đã có thì return True ngay
            if data_path.exists() and len(list(data_path.glob("*.csv"))) > 0:
                return True 
            
            # 2. Nếu chưa có, bắt đầu quy trình tải
            st.info("📥 Đang download dữ liệu từ Kaggle... (Vui lòng đợi)")
            
            # Tạo folder data nếu chưa có
            data_path.mkdir(parents=True, exist_ok=True)
            
            import kagglehub
            
            # Download về cache hệ thống
            cache_path = kagglehub.dataset_download(KaggleDownloader.DATASET_ID)
            
            # Copy từ cache sang folder data/ (Sử dụng shutil)
            source_dir = Path(cache_path)
            copied_count = 0
            for file_path in source_dir.glob("*.csv"):
                shutil.copy(file_path, data_path / file_path.name)
                copied_count += 1
            
            if copied_count > 0:
                st.success("✅ Download và cấu hình thành công! Đang làm mới ứng dụng...")
                
                st.cache_data.clear()
                st.rerun()
                return True
            else:
                st.warning("⚠️ Đã tải nhưng không tìm thấy file .csv.")
                return False
            
        except ImportError:
            st.error("❌ Thiếu thư viện. Chạy: pip install kagglehub")
            return False
        except Exception as e:
            st.error(f"❌ Lỗi download: {str(e)}")
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