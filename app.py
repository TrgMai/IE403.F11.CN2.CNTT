"""
Hệ thống Bán lẻ Thông minh (Retail Smart System)
Ứng dụng Streamlit tích hợp 2 chế độ: Lab (Phòng Thí Nghiệm) và Business App (Ứng Dụng Thực Tế)

Chạy: streamlit run app.py
"""

import streamlit as st
from src.ui.lab_view import show_lab_page
from src.ui.business_view import show_business_page
from src.data_layer import get_data_layer


def show_settings():
    """Hiển thị panel cài đặt dataset."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Cài đặt Dataset")
    
    # Khởi tạo session_state
    if 'dataset_mode' not in st.session_state:
        st.session_state.dataset_mode = "custom"
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = 30000
    
    # Chế độ lấy dữ liệu
    dataset_mode = st.sidebar.radio(
        "Chế độ Dữ liệu:",
        ["Custom (Lấy mẫu)", "Full (Lấy hết)"],
        index=0 if st.session_state.dataset_mode == "custom" else 1,
        help="Custom: lấy số lượng cụ thể | Full: lấy toàn bộ dữ liệu"
    )
    
    # Cập nhật session_state
    st.session_state.dataset_mode = "custom" if "Custom" in dataset_mode else "full"
    
    # Nếu Custom, cho phép chọn số lượng
    if "Custom" in dataset_mode:
        st.session_state.sample_size = st.sidebar.slider(
            "Số records:",
            min_value=1000,
            max_value=500000,
            value=st.session_state.sample_size,
            step=10000,
            help="Số dòng dữ liệu sẽ tải từ transaction_data.csv"
        )
        st.sidebar.info(
            f"**Mode:** Custom\n\n"
            f"**Số records:** {st.session_state.sample_size:,}\n\n"
            "Gợi ý: 30,000 = cân bằng tốc độ & chính xác"
        )
    else:
        st.sidebar.warning(
            "**Mode:** Full Dataset\n\n"
            "Sẽ tải toàn bộ ~2.5M records\n\n"
            "Xử lý sẽ chậm hơn!"
        )
    
    st.sidebar.markdown("---")


def main():
    """
    Hàm chính - Điểm nhập (Entry Point) của ứng dụng.
    Quản lý navigation giữa Lab và Business App.
    """
    
    # Cấu hình trang
    st.set_page_config(
        page_title="Hệ thống Bán lẻ Thông minh",
        page_icon="🏪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS tùy chỉnh - Light Theme
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #ffffff;
        color: #262730;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    /* Typography */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    h2 {
        color: #1f77b4;
        font-weight: 600;
        margin-top: 15px;
    }
    
    h3 {
        color: #2a5c8c;
        font-weight: 500;
    }
    
    /* Button styling */
    button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Input fields */
    input, textarea {
        border-radius: 6px;
        border: 1px solid #d0d7de;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #f6f8fb;
        border-radius: 8px;
        padding: 15px;
    }
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
