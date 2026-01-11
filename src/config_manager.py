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
            
            # ✅ Cập nhật session_state từ overrides
            if "DATASET" in overrides:
                if "mode" in overrides["DATASET"]:
                    st.session_state.dataset_mode = overrides["DATASET"]["mode"]
                if "sample_size" in overrides["DATASET"]:
                    st.session_state.sample_size = overrides["DATASET"]["sample_size"]
            
            # ✅ QUAN TRỌNG: Clear cache để load dữ liệu mới
            st.cache_data.clear()
            
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
            
            # ✅ Reset session_state về config.py defaults
            st.session_state.dataset_mode = DATASET_MODE
            st.session_state.sample_size = DATASET_SAMPLE_SIZE
            
            # ✅ Clear cache
            st.cache_data.clear()
            
            return True
        except Exception as e:
            st.error(f"❌ Lỗi reset config: {str(e)}")
            return False


def show_config_editor():
    """Hiển thị popup chỉnh sửa config với giao diện cải tiến."""
    
    # --- 1. SIDEBAR: CONTROL PANEL ---
    with st.sidebar.container():
        st.markdown("### 🛠️ Control Panel")
        st.caption("Quản lý tham số hệ thống")
        
        # Chia 2 cột đều nhau, khoảng cách nhỏ
        col1, col2 = st.sidebar.columns([1, 1], gap="small")
        
        with col1:
            # Nút Cấu hình: Màu nổi (Primary)
            if st.button("⚙️ Thiết lập", key="open_config", type="primary", use_container_width=True):
                st.session_state.config_popup = True
        
        with col2:
            # Nút Reset: Màu thường (Secondary)
            if st.button("🔄 Mặc định", key="reset_config", use_container_width=True, help="Khôi phục cài đặt gốc"):
                if ConfigManager.reset_overrides():
                    st.toast("✅ Đã khôi phục cấu hình mặc định!", icon="🎉")
                    st.rerun()

    # --- 2. POPUP (MODAL) ---
    if st.session_state.get("config_popup", False):
        with st.container():
            st.divider()
            st.markdown("### ⚙️ Chỉnh sửa Cấu hình")
            
            config = ConfigManager.get_current_config()
            changes = {}
            
            # Tạo Tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Dữ liệu", "🎯 Thuật toán", "🔍 Nâng cao", "ℹ️ Hướng dẫn"]
            )
            
            # === TAB 1: DATASET & CACHE ===
            with tab1:
                st.caption("Cấu hình Dataset và Bộ nhớ đệm")
                
                # Dataset Section
                c1, c2 = st.columns(2)
                with c1:
                    dataset_mode = st.radio(
                        "Chế độ Dataset:",
                        ["custom", "full"],
                        index=0 if config["DATASET"]["mode"] == "custom" else 1,
                        horizontal=True,
                        help="custom: Lấy mẫu | full: Lấy toàn bộ"
                    )
                    if dataset_mode != config["DATASET"]["mode"]:
                        if "DATASET" not in changes: changes["DATASET"] = {}
                        changes["DATASET"]["mode"] = dataset_mode
                
                with c2:
                    sample_size = st.number_input(
                        "Số lượng mẫu (Sample Size):",
                        min_value=1000, max_value=500000,
                        value=config["DATASET"]["sample_size"],
                        step=10000,
                        disabled=(dataset_mode == "full")
                    )
                    if sample_size != config["DATASET"]["sample_size"]:
                        if "DATASET" not in changes: changes["DATASET"] = {}
                        changes["DATASET"]["sample_size"] = sample_size
                
                st.divider()
                
                # Cache Section
                c3, c4 = st.columns([1, 2])
                with c3:
                    st.write("") # Spacer
                    cache_enabled = st.toggle(
                        "Bật Cache",
                        value=config["CACHE"]["enabled"]
                    )
                    if cache_enabled != config["CACHE"]["enabled"]:
                        if "CACHE" not in changes: changes["CACHE"] = {}
                        changes["CACHE"]["enabled"] = cache_enabled
                
                with c4:
                    cache_ttl = st.slider(
                        "Thời gian lưu Cache (giây):",
                        min_value=300, max_value=86400,
                        value=config["CACHE"]["ttl"],
                        step=300
                    )
                    if cache_ttl != config["CACHE"]["ttl"]:
                        if "CACHE" not in changes: changes["CACHE"] = {}
                        changes["CACHE"]["ttl"] = cache_ttl

            # === TAB 2: ALGORITHMS ===
            with tab2:
                st.caption("Tham số các thuật toán chính")
                
                # Apriori
                st.markdown("##### 🛒 Apriori")
                ap_col1, ap_col2, ap_col3 = st.columns(3)
                with ap_col1:
                    min_sup = st.slider("Min Support", 0.0001, 0.1, config["APRIORI"]["min_support"], 0.0001, format="%.4f")
                    if min_sup != config["APRIORI"]["min_support"]:
                        if "APRIORI" not in changes: changes["APRIORI"] = {}
                        changes["APRIORI"]["min_support"] = min_sup
                with ap_col2:
                    min_conf = st.slider("Min Confidence", 0.1, 1.0, config["APRIORI"]["min_confidence"], 0.05)
                    if min_conf != config["APRIORI"]["min_confidence"]:
                        if "APRIORI" not in changes: changes["APRIORI"] = {}
                        changes["APRIORI"]["min_confidence"] = min_conf
                with ap_col3:
                    ap_sam = st.number_input("Apriori Sample", 1000, 100000, config["APRIORI"]["sample_size"], 5000)
                    if ap_sam != config["APRIORI"]["sample_size"]:
                        if "APRIORI" not in changes: changes["APRIORI"] = {}
                        changes["APRIORI"]["sample_size"] = ap_sam

                st.markdown("---")
                
                # K-Means
                st.markdown("##### 🎯 K-Means")
                km_col1, km_col2, km_col3 = st.columns(3)
                with km_col1:
                    n_clus = st.slider("Số Clusters (K)", 2, 10, config["KMEANS"]["n_clusters"])
                    if n_clus != config["KMEANS"]["n_clusters"]:
                        if "KMEANS" not in changes: changes["KMEANS"] = {}
                        changes["KMEANS"]["n_clusters"] = n_clus
                with km_col2:
                    rnd_st = st.number_input("Random State", 0, value=config["KMEANS"]["random_state"])
                    if rnd_st != config["KMEANS"]["random_state"]:
                        if "KMEANS" not in changes: changes["KMEANS"] = {}
                        changes["KMEANS"]["random_state"] = rnd_st
                with km_col3:
                    km_sam = st.number_input("KMeans Sample", 1000, 50000, config["KMEANS"]["sample_size"], 1000)
                    if km_sam != config["KMEANS"]["sample_size"]:
                        if "KMEANS" not in changes: changes["KMEANS"] = {}
                        changes["KMEANS"]["sample_size"] = km_sam

                st.markdown("---")

                # Decision Tree
                st.markdown("##### 🌳 Decision Tree")
                dt_col1, dt_col2, dt_col3 = st.columns(3)
                with dt_col1:
                    max_d = st.slider("Max Depth", 3, 20, config["DECISION_TREE"]["max_depth"])
                    if max_d != config["DECISION_TREE"]["max_depth"]:
                        if "DECISION_TREE" not in changes: changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["max_depth"] = max_d
                with dt_col2:
                    min_s = st.slider("Min Samples Split", 2, 50, config["DECISION_TREE"]["min_samples_split"])
                    if min_s != config["DECISION_TREE"]["min_samples_split"]:
                        if "DECISION_TREE" not in changes: changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["min_samples_split"] = min_s
                with dt_col3:
                    dt_sam = st.number_input("DT Sample", 1000, 100000, config["DECISION_TREE"]["sample_size"], 5000)
                    if dt_sam != config["DECISION_TREE"]["sample_size"]:
                        if "DECISION_TREE" not in changes: changes["DECISION_TREE"] = {}
                        changes["DECISION_TREE"]["sample_size"] = dt_sam

            # === TAB 3: DETAILS (Naive Bayes & Rough Set) ===
            with tab3:
                st.caption("Cấu hình nâng cao")
                
                # Naive Bayes
                st.subheader("Naive Bayes")
                nb_col1, nb_col2 = st.columns(2)
                with nb_col1:
                    laplace = st.toggle("Laplace Smoothing", value=config["NAIVE_BAYES"]["laplace_smoothing"])
                    if laplace != config["NAIVE_BAYES"]["laplace_smoothing"]:
                        if "NAIVE_BAYES" not in changes: changes["NAIVE_BAYES"] = {}
                        changes["NAIVE_BAYES"]["laplace_smoothing"] = laplace
                with nb_col2:
                    nb_sam = st.number_input("NB Sample Size", 1000, 100000, config["NAIVE_BAYES"]["sample_size"], 5000)
                    if nb_sam != config["NAIVE_BAYES"]["sample_size"]:
                        if "NAIVE_BAYES" not in changes: changes["NAIVE_BAYES"] = {}
                        changes["NAIVE_BAYES"]["sample_size"] = nb_sam
                
                st.divider()
                
                # Rough Set
                st.subheader("Rough Set")
                rs_col1, rs_col2 = st.columns(2)
                with rs_col1:
                    max_f = st.slider("Max Features", 2, 20, config["ROUGH_SET"]["max_features"])
                    if max_f != config["ROUGH_SET"]["max_features"]:
                        if "ROUGH_SET" not in changes: changes["ROUGH_SET"] = {}
                        changes["ROUGH_SET"]["max_features"] = max_f
                with rs_col2:
                    rs_sam = st.number_input("RS Sample Size", 1000, 100000, config["ROUGH_SET"]["sample_size"], 5000)
                    if rs_sam != config["ROUGH_SET"]["sample_size"]:
                        if "ROUGH_SET" not in changes: changes["ROUGH_SET"] = {}
                        changes["ROUGH_SET"]["sample_size"] = rs_sam

            # === TAB 4: INFO ===
            with tab4:
                st.info("""
                **Hướng dẫn:**
                1. Thay đổi các tham số ở các tab bên cạnh.
                2. Nhấn **"Lưu"** để áp dụng (có hiệu lực ngay lần chạy tới).
                3. Nhấn **"Reset"** ở thanh bên trái nếu muốn quay về mặc định.
                
                **Ghi chú:**
                - *Sample Size:* Số lượng dòng dữ liệu dùng để huấn luyện mô hình (giảm nếu chạy chậm).
                - *Cache:* Nên bật để tăng tốc độ tải trang.
                """)

            # --- 3. FOOTER BUTTONS ---
            st.divider()
            
            f_col1, f_col2, f_col3 = st.columns([3, 1, 1])
            
            with f_col2:
                # Nút Lưu
                if st.button("💾 Lưu", key="save_config", type="primary", use_container_width=True):
                    if changes:
                        if ConfigManager.save_overrides(changes):
                            st.toast("✅ Đã lưu cấu hình mới!", icon="💾")
                            st.session_state.config_popup = False
                            st.rerun()
                    else:
                        st.toast("ℹ️ Không có thay đổi nào để lưu.", icon="✅")
            
            with f_col3:
                # Nút Đóng
                if st.button("❌ Đóng", key="close_config", use_container_width=True):
                    st.session_state.config_popup = False
                    st.rerun()
