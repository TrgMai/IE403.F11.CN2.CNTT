"""
Configuration Module - Cấu hình chung cho ứng dụng.
Quản lý các tham số như sample_size, algorithm parameters, etc.
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Chế độ lấy dữ liệu: "custom" (lấy số lượng) hoặc "full" (lấy hết)
DATASET_MODE = "custom"  # "custom" hoặc "full"
DATASET_SAMPLE_SIZE = 30000  # Số records khi mode = "custom"

# ============================================================================
# ALGORITHM PARAMETERS
# ============================================================================

# APRIORI - Association Rule Mining
APRIORI_CONFIG = {
    "min_support": 0.005,      # 0.5% - Tăng từ 0.1% để giảm memory
    "min_confidence": 0.3,     # 30% - Rule phải có độ tin cậy ít nhất 30%
    "sample_size": 10000       # Giảm từ 30000 để tiết kiệm memory
}

# K-MEANS - Customer Segmentation (RFM)
KMEANS_CONFIG = {
    "n_clusters": 3,           # 3 cụm: High/Medium/Low value
    "random_state": 42,        # Seed để tái tạo kết quả
    "sample_size": 10000       # Số customers dùng cho clustering
}

# DECISION TREE - Campaign Response Prediction
DECISION_TREE_CONFIG = {
    "max_depth": 5,            # Độ sâu tối đa
    "min_samples_split": 20,   # Số mẫu tối thiểu để split node
    "random_state": 42,
    "sample_size": 50000       # Dữ liệu huấn luyện
}

# NAIVE BAYES - Classification
NAIVE_BAYES_CONFIG = {
    "laplace_smoothing": True, # Tránh xác suất = 0
    "sample_size": 50000
}

# BAYESIAN NETWORK - Probabilistic Graphical Model
BAYESIAN_NETWORK_CONFIG = {
    "dag_structure": {
        "nodes": ["AGE", "INCOME", "HOMEOWNER"],
        "edges": [("AGE", "INCOME"), ("INCOME", "HOMEOWNER")]
    },
    "sample_size": 10000
}

# ROUGH SET - Feature Selection
ROUGH_SET_CONFIG = {
    "max_features": 5,         # Chọn 5 feature quan trọng nhất
    "sample_size": 50000
}

# ============================================================================
# UI/UX CONFIGURATION
# ============================================================================
APP_TITLE = "Hệ thống Bán lẻ Thông minh"
APP_ICON = "🏪"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Color scheme
PRIMARY_COLOR = "#1f77b4"
SUCCESS_COLOR = "#2ca02c"
WARNING_COLOR = "#ff7f0e"
ERROR_COLOR = "#d62728"

# ============================================================================
# DATA FILE PATHS
# ============================================================================
DATA_DIR = "data/"
DATA_FILES = {
    "transaction": f"{DATA_DIR}transaction_data.csv",
    "product": f"{DATA_DIR}product.csv",
    "demographic": f"{DATA_DIR}hh_demographic.csv",
    "coupon": f"{DATA_DIR}coupon.csv",
    "coupon_redempt": f"{DATA_DIR}coupon_redempt.csv",
    "campaign_desc": f"{DATA_DIR}campaign_desc.csv",
    "campaign_table": f"{DATA_DIR}campaign_table.csv",
    "causal_data": f"{DATA_DIR}causal_data.csv"
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour
