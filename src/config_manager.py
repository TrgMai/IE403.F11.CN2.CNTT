"""
Config Manager - Quản lý và lưu cấu hình từ giao diện.
Cho phép chỉnh sửa config.py trực tiếp từ ứng dụng.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any
import streamlit as st
from src.config import (
    DATASET_MODE, DATASET_SAMPLE_SIZE,
    APRIORI_CONFIG, KMEANS_CONFIG, DECISION_TREE_CONFIG,
    NAIVE_BAYES_CONFIG, BAYESIAN_NETWORK_CONFIG, ROUGH_SET_CONFIG,
    CACHE_ENABLED, CACHE_TTL
)


class ConfigManager:
    """Quản lý cấu hình ứng dụng."""
    
    CONFIG_FILE = "config_overrides.json"  # File lưu các override từ giao diện
    
    @staticmethod
    def get_current_config() -> Dict[str, Any]:
        """Lấy cấu hình hiện tại (bao gồm overrides nếu có)."""
        config = {
            "DATASET": {
                "mode": DATASET_MODE,
                "sample_size": DATASET_SAMPLE_SIZE
            },
            "APRIORI": APRIORI_CONFIG.copy(),
            "KMEANS": KMEANS_CONFIG.copy(),
            "DECISION_TREE": DECISION_TREE_CONFIG.copy(),
            "NAIVE_BAYES": NAIVE_BAYES_CONFIG.copy(),
            "BAYESIAN_NETWORK": BAYESIAN_NETWORK_CONFIG.copy(),
            "ROUGH_SET": ROUGH_SET_CONFIG.copy(),
            "CACHE": {
                "enabled": CACHE_ENABLED,
                "ttl": CACHE_TTL
            }
        }
        
        # Tải overrides từ file nếu có
        if os.path.exists(ConfigManager.CONFIG_FILE):
            try:
                with open(ConfigManager.CONFIG_FILE, 'r') as f:
                    overrides = json.load(f)
                    # Merge overrides vào config
                    for key, value in overrides.items():
                        if key in config:
                            if isinstance(config[key], dict):
                                config[key].update(value)
                            else:
                                config[key] = value
            except Exception as e:
                st.warning(f"⚠️ Lỗi đọc config overrides: {str(e)}")
        
        return config
    
    @staticmethod
    def save_overrides(overrides: Dict[str, Any]) -> bool:
        """Lưu các thay đổi config vào file."""
        try:
            with open(ConfigManager.CONFIG_FILE, 'w') as f:
                json.dump(overrides, f, indent=2)
            return True
        except Exception as e:
            st.error(f"❌ Lỗi lưu config: {str(e)}")
            return False
    
    @staticmethod
    def reset_overrides() -> bool:
        """Xóa tất cả overrides về mặc định."""
        try:
            if os.path.exists(ConfigManager.CONFIG_FILE):
                os.remove(ConfigManager.CONFIG_FILE)
            return True
        except Exception as e:
            st.error(f"❌ Lỗi reset config: {str(e)}")
            return False


def show_config_editor():
    """Hiển thị popup chỉnh sửa config."""
    
    # Tiêu đề Config Manager
    st.sidebar.markdown("### ⚙️ Quản lý Cấu hình")
    st.sidebar.caption("Chỉnh sửa tham số thuật toán & dữ liệu")
    
    # Nút mở popup
    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    
    with col1:
        if st.button("⚙️ Cấu hình", key="open_config"):
            st.session_state.config_popup = True
    
    with col2:
        if st.button("🔄 Reset", key="reset_config"):
            if ConfigManager.reset_overrides():
                st.success("✅ Reset cấu hình thành công!")
                st.rerun()
    
    # Hiển thị popup (modal)
    if st.session_state.get("config_popup", False):
        with st.container():
            st.markdown("---")
            st.markdown("## ⚙️ Chỉnh sửa Cấu hình")
            
            config = ConfigManager.get_current_config()
            changes = {}
            
            # Tab cho từng phần
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Dataset & Cache", "🎯 Thuật toán", "🔍 Chi tiết", "📝 Thông tin"]
            )
            
            with tab1:
                st.subheader("Dataset Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    dataset_mode = st.radio(
                        "Chế độ Dataset:",
                        ["custom", "full"],
                        index=0 if config["DATASET"]["mode"] == "custom" else 1,
                        help="custom: lấy mẫu | full: lấy toàn bộ"
                    )
                    if dataset_mode != config["DATASET"]["mode"]:
                        if "DATASET" not in changes:
                            changes["DATASET"] = {}
                        changes["DATASET"]["mode"] = dataset_mode
                
                with col2:
                    sample_size = st.number_input(
                        "Sample Size:",
                        min_value=1000,
                        max_value=500000,
                        value=config["DATASET"]["sample_size"],
                        step=10000,
                        help="Số records để tải khi mode=custom"
                    )
                    if sample_size != config["DATASET"]["sample_size"]:
                        if "DATASET" not in changes:
                            changes["DATASET"] = {}
                        changes["DATASET"]["sample_size"] = sample_size
                
                st.markdown("---")
                st.subheader("Cache Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    cache_enabled = st.checkbox(
                        "Bật Cache",
                        value=config["CACHE"]["enabled"],
                        help="Có sử dụng cache hay không"
                    )
                    if cache_enabled != config["CACHE"]["enabled"]:
                        if "CACHE" not in changes:
                            changes["CACHE"] = {}
                        changes["CACHE"]["enabled"] = cache_enabled
                
                with col2:
                    cache_ttl = st.number_input(
                        "Cache TTL (giây):",
                        min_value=300,
                        max_value=86400,
                        value=config["CACHE"]["ttl"],
                        step=300,
                        help="Thời gian cache tồn tại (giây)"
                    )
                    if cache_ttl != config["CACHE"]["ttl"]:
                        if "CACHE" not in changes:
                            changes["CACHE"] = {}
                        changes["CACHE"]["ttl"] = cache_ttl
            
            with tab2:
                st.subheader("Apriori Configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_support = st.slider(
                        "Min Support",
                        min_value=0.0001,
                        max_value=0.1,
                        value=config["APRIORI"]["min_support"],
                        step=0.0001,
                        format="%.4f",
                        help="Ngưỡng hỗ trợ tối thiểu"
                    )
                    if min_support != config["APRIORI"]["min_support"]:
                        if "APRIORI" not in changes:
                            changes["APRIORI"] = {}
                        changes["APRIORI"]["min_support"] = min_support
                
                with col2:
                    min_confidence = st.slider(
                        "Min Confidence",
                        min_value=0.1,
                        max_value=1.0,
                        value=config["APRIORI"]["min_confidence"],
                        step=0.05,
                        format="%.2f",
                        help="Ngưỡng độ tin cậy tối thiểu"
                    )
                    if min_confidence != config["APRIORI"]["min_confidence"]:
                        if "APRIORI" not in changes:
                            changes["APRIORI"] = {}
                        changes["APRIORI"]["min_confidence"] = min_confidence
                
                with col3:
                    apriori_sample = st.number_input(
                        "Apriori Sample Size",
                        min_value=1000,
                        max_value=100000,
                        value=config["APRIORI"]["sample_size"],
                        step=5000
                    )
                    if apriori_sample != config["APRIORI"]["sample_size"]:
                        if "APRIORI" not in changes:
                            changes["APRIORI"] = {}
                        changes["APRIORI"]["sample_size"] = apriori_sample
                
                st.markdown("---")
                st.subheader("K-Means Configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_clusters = st.slider(
                        "Số Clusters",
                        min_value=2,
                        max_value=10,
                        value=config["KMEANS"]["n_clusters"],
                        help="Số cụm khách hàng"
                    )
                    if n_clusters != config["KMEANS"]["n_clusters"]:
                        if "KMEANS" not in changes:
                            changes["KMEANS"] = {}
                        changes["KMEANS"]["n_clusters"] = n_clusters
                
                with col2:
                    random_state = st.number_input(
                        "Random State",
                        min_value=0,
                        value=config["KMEANS"]["random_state"],
                        help="Seed để tái tạo kết quả"
                    )
                    if random_state != config["KMEANS"]["random_state"]:
                        if "KMEANS" not in changes:
                            changes["KMEANS"] = {}
                        changes["KMEANS"]["random_state"] = random_state
                
                with col3:
                    kmeans_sample = st.number_input(
                        "K-Means Sample Size",
                        min_value=1000,
                        max_value=50000,
                        value=config["KMEANS"]["sample_size"],
                        step=1000
                    )
                    if kmeans_sample != config["KMEANS"]["sample_size"]:
                        if "KMEANS" not in changes:
                            changes["KMEANS"] = {}
                        changes["KMEANS"]["sample_size"] = kmeans_sample
                
                st.markdown("---")
                st.subheader("Decision Tree Configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_depth = st.slider(
                        "Max Depth",
                        min_value=3,
                        max_value=20,
                        value=config["DECISION_TREE"]["max_depth"],
                        help="Độ sâu tối đa của cây"
                    )
                    if max_depth != config["DECISION_TREE"]["max_depth"]:
                        if "DECISION_TREE" not in changes:
                            changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["max_depth"] = max_depth
                
                with col2:
                    min_samples = st.slider(
                        "Min Samples Split",
                        min_value=2,
                        max_value=50,
                        value=config["DECISION_TREE"]["min_samples_split"],
                        help="Số mẫu tối thiểu để tách node"
                    )
                    if min_samples != config["DECISION_TREE"]["min_samples_split"]:
                        if "DECISION_TREE" not in changes:
                            changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["min_samples_split"] = min_samples
                
                with col3:
                    dt_sample = st.number_input(
                        "DT Sample Size",
                        min_value=1000,
                        max_value=100000,
                        value=config["DECISION_TREE"]["sample_size"],
                        step=5000
                    )
                    if dt_sample != config["DECISION_TREE"]["sample_size"]:
                        if "DECISION_TREE" not in changes:
                            changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["sample_size"] = dt_sample
            
            with tab3:
                st.subheader("Naive Bayes Configuration")
                laplace = st.checkbox(
                    "Laplace Smoothing",
                    value=config["NAIVE_BAYES"]["laplace_smoothing"],
                    help="Sử dụng Laplace Smoothing"
                )
                if laplace != config["NAIVE_BAYES"]["laplace_smoothing"]:
                    if "NAIVE_BAYES" not in changes:
                        changes["NAIVE_BAYES"] = {}
                    changes["NAIVE_BAYES"]["laplace_smoothing"] = laplace
                
                nb_sample = st.number_input(
                    "Naive Bayes Sample Size",
                    min_value=1000,
                    max_value=100000,
                    value=config["NAIVE_BAYES"]["sample_size"],
                    step=5000
                )
                if nb_sample != config["NAIVE_BAYES"]["sample_size"]:
                    if "NAIVE_BAYES" not in changes:
                        changes["NAIVE_BAYES"] = {}
                    changes["NAIVE_BAYES"]["sample_size"] = nb_sample
                
                st.markdown("---")
                st.subheader("Rough Set Configuration")
                max_features = st.slider(
                    "Max Features",
                    min_value=2,
                    max_value=20,
                    value=config["ROUGH_SET"]["max_features"],
                    help="Số feature tối đa để chọn"
                )
                if max_features != config["ROUGH_SET"]["max_features"]:
                    if "ROUGH_SET" not in changes:
                        changes["ROUGH_SET"] = {}
                    changes["ROUGH_SET"]["max_features"] = max_features
                
                rs_sample = st.number_input(
                    "Rough Set Sample Size",
                    min_value=1000,
                    max_value=100000,
                    value=config["ROUGH_SET"]["sample_size"],
                    step=5000
                )
                if rs_sample != config["ROUGH_SET"]["sample_size"]:
                    if "ROUGH_SET" not in changes:
                        changes["ROUGH_SET"] = {}
                    changes["ROUGH_SET"]["sample_size"] = rs_sample
            
            with tab4:
                st.info("""
                ### 📝 Hướng dẫn sử dụng Config Editor
                
                **Các tùy chỉnh:**
                - 📊 **Dataset & Cache:** Cấu hình dữ liệu và bộ nhớ cache
                - 🎯 **Thuật toán:** Chỉnh các tham số chính của từng thuật toán
                - 🔍 **Chi tiết:** Cấu hình chi tiết cho Naive Bayes, Rough Set
                
                **Lưu ý:**
                - Thay đổi được lưu vào file `config_overrides.json`
                - Bấm "🔄 Reset" để quay lại cấu hình mặc định
                - Các thay đổi sẽ có hiệu lực ngay trên lần chạy tiếp theo
                
                **Default Config:**
                - `DATASET_MODE`: custom
                - `DATASET_SAMPLE_SIZE`: 30,000
                - `APRIORI_CONFIG`: min_support=0.001, min_confidence=0.3
                - `KMEANS_CONFIG`: n_clusters=3
                - `DECISION_TREE_CONFIG`: max_depth=5
                """)
            
            st.markdown("---")
            
            # Nút Save và Close
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write("")  # Spacing
            
            with col2:
                if st.button("💾 Lưu", key="save_config", use_container_width=True):
                    if changes:
                        if ConfigManager.save_overrides(changes):
                            st.success("✅ Lưu cấu hình thành công!")
                            st.session_state.config_popup = False
                            st.rerun()
                    else:
                        st.info("ℹ️ Không có thay đổi nào")
            
            with col3:
                if st.button("❌ Đóng", key="close_config", use_container_width=True):
                    st.session_state.config_popup = False
                    st.rerun()
