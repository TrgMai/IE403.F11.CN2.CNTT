"""
Hệ thống Bán lẻ Thông minh (Retail Smart System)
Ứng dụng Streamlit tích hợp 2 chế độ: Lab (Phòng Thí Nghiệm) và Business App (Ứng Dụng Thực Tế)

Chạy: streamlit run app.py
"""

import streamlit as st
from src.ui.lab_view import show_lab_page
from src.ui.business_view import show_business_page
from src.data_layer import get_data_layer
from src.config import (
    DATASET_MODE, DATASET_SAMPLE_SIZE,
    APP_TITLE, APP_ICON, APP_LAYOUT, SIDEBAR_STATE,
    PRIMARY_COLOR
)
from src.config_manager import show_config_editor


def show_settings():
    """Hiển thị panel config editor (Dataset cài đặt đã có trong Config Editor)."""
    st.sidebar.markdown("---")
    
    # Khởi tạo session_state từ config.py (sử dụng config editor để chỉnh)
    if 'dataset_mode' not in st.session_state:
        st.session_state.dataset_mode = DATASET_MODE
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = DATASET_SAMPLE_SIZE
    if 'config_popup' not in st.session_state:
        st.session_state.config_popup = False
    show_config_editor()
    st.sidebar.markdown("---")


def main():
    """
    Hàm chính - Điểm nhập (Entry Point) của ứng dụng.
    Quản lý navigation giữa Lab và Business App.
    """
    
    # Cấu hình trang từ config.py
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    
    # CSS tùy chỉnh - Light Theme (từ config.py)
    st.markdown(f"""
    <style>
    /* Main styling */
    .main {{
        background-color: #ffffff;
        color: #262730;
    }}
    
    .sidebar .sidebar-content {{
        background-color: #f0f2f6;
    }}
    
    /* Typography */
    h1 {{
        color: {PRIMARY_COLOR};
        font-weight: 700;
        margin-bottom: 20px;
    }}
    
    h2 {{
        color: {PRIMARY_COLOR};
        font-weight: 600;
        margin-top: 15px;
    }}
    
    h3 {{
        color: #2a5c8c;
        font-weight: 500;
    }}
    
    /* Button styling */
    button {{
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    /* Input fields */
    input, textarea {{
        border-radius: 6px;
        border: 1px solid #d0d7de;
    }}
    
    /* Metrics */
    [data-testid="metric-container"] {{
        background-color: #f6f8fb;
        border-radius: 8px;
        padding: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Cài đặt Dataset
    show_settings()
    
    st.sidebar.title("Retail Smart System")
    
    mode = st.sidebar.radio(
        "Chế độ:",
        ["Phòng Thí Nghiệm", "Ứng Dụng Thực Tế"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Chuyển đổi giữa hai chế độ
    if mode == "Phòng Thí Nghiệm":
        show_lab_page()
    else:
        show_business_page()


if __name__ == "__main__":
    main()
