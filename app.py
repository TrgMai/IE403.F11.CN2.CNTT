"""
Hệ thống Bán lẻ Thông minh (Retail Smart System)
Ứng dụng Streamlit tích hợp 2 chế độ: Lab (Phòng Thí Nghiệm) và Business App (Ứng Dụng Thực Tế)

Chạy: streamlit run app.py
"""

import streamlit as st
from src.ui.lab_view import show_lab_page
from src.ui.business_view import show_business_page
from src.config_manager import show_config_editor, ConfigManager
from src.config import (
    APP_TITLE, APP_ICON, APP_LAYOUT, SIDEBAR_STATE,
    PRIMARY_COLOR
)

def init_session_state():
    """Khởi tạo các biến Session State từ ConfigManager."""
    # Lấy config hiện tại (đã bao gồm overrides nếu có)
    current_config = ConfigManager.get_current_config()
    
    # Chỉ khởi tạo nếu chưa có trong session_state
    if 'dataset_mode' not in st.session_state:
        st.session_state.dataset_mode = current_config["DATASET"]["mode"]
        
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = current_config["DATASET"]["sample_size"]
        
    if 'config_popup' not in st.session_state:
        st.session_state.config_popup = False

def apply_custom_css():
    """Áp dụng CSS tùy chỉnh cho giao diện."""
    st.markdown(f"""
    <style>
    /* Main styling */
    .main {{
        background-color: #ffffff;
        color: #262730;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #f0f2f6;
        border-right: 1px solid #dcdcdc;
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
    
    /* Button styling - hover effect */
    button {{
        border-radius: 6px;
        transition: all 0.3s ease;
    }}
    
    /* Metric Cards */
    [data-testid="metric-container"] {{
        background-color: #f6f8fb;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    """
    Hàm chính - Điểm nhập (Entry Point) của ứng dụng.
    Quản lý navigation giữa Lab và Business App.
    """
    
    # 1. Cấu hình trang (Phải gọi đầu tiên)
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    
    # 2. Khởi tạo State và CSS
    init_session_state()
    apply_custom_css()
    
    # 3. Xây dựng Sidebar (Navigation & Settings)
    with st.sidebar:
        # Tiêu đề ứng dụng
        st.title("🛍️ Retail Smart System")

        st.divider()
        
        # Navigation (Menu chọn chế độ)
        mode = st.radio(
            "Chọn chế độ làm việc:",
            ["Phòng Thí Nghiệm", "Ứng Dụng Thực Tế"],
            index=0,
            key="app_mode_selection"
        )
        
        st.divider()
        
        # Control Panel (Nút Cấu hình từ ConfigManager)
        show_config_editor()
        
        # Footer thông tin (Optional)
        st.markdown("---")
        st.caption("© 2024 Retail Analytics")

    # 4. Điều hướng nội dung chính
    if mode == "Phòng Thí Nghiệm":
        show_lab_page()
    else:
        show_business_page()

if __name__ == "__main__":
    main()