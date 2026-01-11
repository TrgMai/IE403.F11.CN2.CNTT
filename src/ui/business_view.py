"""
Giao diện Ứng Dụng Thực Tế (Business App View) - Bảng điều khiển cho phân tích kinh doanh.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_layer import get_data_layer
from src.data_preprocessing import DataPreprocessor
from src.algorithms.kmeans import KMeansClustering
from src.algorithms.decision_tree import DecisionTreeCART
from src.algorithms.apriori import AprioriAlgorithm
from src.config import APRIORI_CONFIG, KMEANS_CONFIG, DECISION_TREE_CONFIG


def show_data_analysis():
    """Hiển thị tính năng Phân tích Dữ liệu."""
    st.header("📊 Phân tích Dữ liệu")
    st.write("Thống kê chi tiết về dữ liệu giao dịch, sản phẩm và khách hàng")
    
    if st.button("🔬 Phân tích Dữ liệu", key="data_analysis"):
        with st.spinner("Đang phân tích dữ liệu..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data(sample_size=None)  # Lấy toàn bộ
            product_df = data_layer.load_product_data()
            
            preprocessor = DataPreprocessor()
            analysis = preprocessor.analyze_data(trans_df)
            
            if not analysis:
                st.error("Không thể phân tích dữ liệu.")
                return
            
            # Hiển thị tóm tắt chính
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Tổng Records", f"{analysis['total_rows']:,}")
            with col2:
                st.metric("📋 Tổng Cột", analysis['total_columns'])
            with col3:
                st.metric("💾 Dung lượng", f"{analysis['memory_usage_mb']:.2f} MB")
            with col4:
                st.metric("🔄 Hàng Trùng", analysis['duplicate_rows'])
            
            # Chi tiết từng cột
            st.subheader("📋 Chi tiết Cột (Columns)")
            col_info_list = []
            for col_name, info in analysis['column_info'].items():
                col_info_list.append({
                    'Cột': col_name,
                    'Kiểu dữ liệu': info['dtype'],
                    'Unique': f"{info['unique']:,}",
                    'Null': f"{info['null_count']:,}",
                    'Null %': f"{info['null_pct']:.2f}%"
                })
            
            col_df = pd.DataFrame(col_info_list)
            st.dataframe(col_df, use_container_width=True)
            
            st.success("✅ Phân tích hoàn tất!")


def show_data_preprocessing():
    """Hiển thị tính năng Tiền xử lý Dữ liệu."""
    st.header("🔧 Tiền xử lý Dữ liệu")
    st.write("Làm sạch dữ liệu: loại bỏ null, duplicates, và các bất thường")
    
    col1, col2 = st.columns(2)
    with col1:
        remove_nulls = st.checkbox("Loại bỏ NULL values", value=True)
    with col2:
        remove_duplicates = st.checkbox("Loại bỏ hàng trùng", value=True)
    
    if st.button("🔄 Tiền xử lý Dữ liệu", key="data_preprocessing"):
        with st.spinner("Đang tiền xử lý..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data(sample_size=None)
            
            preprocessor = DataPreprocessor()
            
            # Phân tích trước xử lý
            st.subheader("📊 Trước Xử lý")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🔹 Tổng hàng: {len(trans_df):,}")
            with col2:
                st.info(f"🔹 Hàng Trùng: {trans_df.duplicated().sum():,}")
            
            # Tiền xử lý
            processed_df, preprocessing_info = preprocessor.preprocess_data(
                trans_df, 
                remove_nulls=remove_nulls,
                remove_duplicates=remove_duplicates
            )
            
            # Kết quả sau xử lý
            st.subheader("📊 Sau Xử lý")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ Hàng còn lại: {len(processed_df):,}")
            with col2:
                st.info(f"🔹 Hàng bị loại: {preprocessing_info['duplicates_removed'] + preprocessing_info['nulls_removed']:,}")
            with col3:
                st.metric("📈 Data Retention", f"{preprocessing_info['data_retention_pct']:.2f}%")
            
            # Chi tiết xử lý
            st.subheader("📋 Chi tiết Tiền xử lý")
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.write("**Hàng bị loại bỏ:**")
                st.write(f"- Duplicates: {preprocessing_info['duplicates_removed']:,}")
                st.write(f"- Nulls: {preprocessing_info['nulls_removed']:,}")
            
            with detail_cols[1]:
                st.write("**Cột bị xóa (toàn NULL):**")
                if preprocessing_info['all_null_cols_removed']:
                    for col in preprocessing_info['all_null_cols_removed']:
                        st.write(f"- {col}")
                else:
                    st.write("- Không có cột nào")
            
            st.success("✅ Tiền xử lý hoàn tất!")


def show_customer_segmentation():
    """Hiển thị tính năng Phân khúc Khách hàng."""
    st.header("👥 Phân khúc Khách hàng")
    st.write("Sử dụng k-Means Clustering để chia khách hàng thành các nhóm có tính chất tương tự")
    
    n_clusters = st.slider("Số nhóm khách hàng", 2, 5, KMEANS_CONFIG['n_clusters'])
    
    if st.button("🔄 Phân tích Khách hàng", key="seg_analyze"):
        with st.spinner("Đang phân tích..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data()
            
            if len(trans_df) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            kmeans = KMeansClustering(n_clusters=n_clusters)
            result = kmeans.fit(trans_df)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            rfm_data = result['rfm_data']
            
            # Thống kê chung
            st.subheader("📊 Thống kê Khách hàng")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng khách hàng", len(rfm_data))
            with col2:
                st.metric("Trung bình Recency", f"{rfm_data['Recency'].mean():.1f} ngày")
            with col3:
                st.metric("Tổng doanh thu", f"${rfm_data['Monetary'].sum():,.0f}")
            
            # Phân tích từng nhóm
            st.subheader("🎯 Chi tiết từng Nhóm")
            
            cluster_strategies = {
                0: "**Chiến lược:** Giữ chân - Khách hàng có giá trị cao, cần chương trình loyalty",
                1: "**Chiến lược:** Kích hoạt lại - Khách hàng cũ, cần chiến dịch re-engagement",
                2: "**Chiến lược:** Phát triển - Khách hàng mới hoặc tiềm năng, cần hỗ trợ",
                3: "**Chiến lược:** Quản lý - Khách hàng trung bình, tối ưu hóa chi phí",
                4: "**Chiến lược:** Phục vụ - Khách hàng đa dạng, cần chiến lược đa chiều"
            }
            
            for cluster_id in range(n_clusters):
                with st.expander(f"📌 Nhóm {cluster_id + 1}", expanded=(cluster_id == 0)):
                    cluster_data = rfm_data[rfm_data['Cluster'] == cluster_id]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Số khách", len(cluster_data))
                    with col2:
                        st.metric("% tổng", f"{len(cluster_data)/len(rfm_data)*100:.1f}%")
                    with col3:
                        st.metric("Avg Spending", f"${cluster_data['Monetary'].mean():.0f}")
                    
                    st.write(cluster_strategies.get(cluster_id, ""))
                    
                    # Chi tiết RFM
                    rfm_avg = cluster_data[['Recency', 'Frequency', 'Monetary']].mean()
                    st.info(f"""
                    **RFM Profile:**
                    - Recency (ngày gần đây): {rfm_avg['Recency']:.1f}
                    - Frequency (lần mua): {rfm_avg['Frequency']:.1f}
                    - Monetary (chi tiêu): ${rfm_avg['Monetary']:.0f}
                    """)
            
            # Biểu đồ phân bố
            st.subheader("📈 Biểu đồ Phân bố")
            
            # Pie chart - Phân bố khách hàng
            cluster_counts = rfm_data['Cluster'].value_counts().sort_index()
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"Nhóm {i+1}" for i in cluster_counts.index],
                values=cluster_counts.values,
                textposition='inside',
                textinfo='label+percent'
            )])
            fig_pie.update_layout(title="Phân bố Khách hàng theo Nhóm", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Scatter 3D RFM
            fig_3d = px.scatter_3d(rfm_data,
                                   x='Recency', y='Frequency', z='Monetary',
                                   color='Cluster',
                                   title='Phân bố 3D RFM theo Nhóm',
                                   labels={'Cluster': 'Nhóm'},
                                   color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_3d, use_container_width=True)


def show_campaign_response_prediction():
    """Hiển thị tính năng Dự đoán Phản hồi Chiến dịch."""
    st.header("📢 Dự đoán Phản hồi Chiến dịch")
    st.write("Sử dụng Decision Tree để dự đoán khách hàng nào sẽ phản hồi chiến dịch")
    
    if st.button("🔮 Dự đoán Phản hồi", key="campaign_predict"):
        with st.spinner("Đang dự đoán..."):
            data_layer = get_data_layer()
            merged = data_layer.get_merged_dataset()
            
            if len(merged) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            # Tạo nhãn mục tiêu (ví dụ: khách hàng mua nhiều = phản hồi chiến dịch)
            merged['CAMPAIGN_RESPONSE'] = (merged['SALES_VALUE'] > 
                                          merged['SALES_VALUE'].median()).astype(int)
            
            model = DecisionTreeCART(max_depth=DECISION_TREE_CONFIG['max_depth'], 
                                    min_samples_split=DECISION_TREE_CONFIG['min_samples_split'])
            result = model.train(merged, target_column='CAMPAIGN_RESPONSE')
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Dự đoán thành công (Accuracy: {result['accuracy']:.2%})")
            
            # Hiển thị kết quả
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Độ chính xác", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Số nút cây", result['num_nodes'])
            with col3:
                st.metric("Độ sâu cây", result['max_depth'])
            
            # Feature Importance
            st.subheader("🎯 Yếu tố Ảnh hưởng tới Phản hồi Chiến dịch")
            features_df = pd.DataFrame(list(result['feature_importance'].items()),
                                      columns=['Yếu tố', 'Mức độ Ảnh hưởng'])
            features_df = features_df.sort_values('Mức độ Ảnh hưởng', ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(x=features_df['Mức độ Ảnh hưởng'],
                      y=features_df['Yếu tố'],
                      orientation='h',
                      marker=dict(color='#FF6B6B'))
            ])
            fig.update_layout(
                title="Tầm quan trọng các Yếu tố",
                xaxis_title="Mức độ Ảnh hưởng",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix - Đánh giá Chất lượng Dự đoán")
            cm = result['confusion_matrix']
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Không phản hồi', 'Phản hồi'],
                y=['Không phản hồi', 'Phản hồi'],
                text=cm,
                texttemplate='%{text}',
                colorscale='RdYlGn'
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Dự đoán",
                yaxis_title="Thực tế",
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)


def show_product_recommendation():
    """Hiển thị tính năng Gợi ý Sản phẩm."""
    st.header("🛍️ Gợi ý Sản phẩm Thông minh")
    st.write("Sử dụng Association Rules (Apriori) để gợi ý sản phẩm liên quan")
    
    # Khởi tạo session_state cho lưu trữ rules
    if 'recom_rules_found' not in st.session_state:
        st.session_state.recom_rules_found = False
        st.session_state.apriori_rules = None
        st.session_state.trans_df = None
        st.session_state.product_df = None
    
    if st.button("🔍 Tìm Luật Kết hợp", key="recom_find"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data()
            product_df = data_layer.load_product_data()
            
            if len(trans_df) == 0 or len(product_df) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            apriori = AprioriAlgorithm()
            itemsets, rules = apriori.run(trans_df, min_support=APRIORI_CONFIG['min_support'], 
                                         min_confidence=APRIORI_CONFIG['min_confidence'])
            
            if len(rules) == 0:
                st.warning("Không tìm thấy Association Rules.")
                st.session_state.recom_rules_found = False
                return
            
            # Lưu vào session_state
            st.session_state.apriori_rules = rules
            st.session_state.trans_df = trans_df
            st.session_state.product_df = product_df
            st.session_state.recom_rules_found = True
            
            st.success(f"✅ Tìm thấy {len(rules)} Luật Kết hợp")
    
    # Hiển thị giao diện chọn sản phẩm nếu đã tìm được rules
    if st.session_state.recom_rules_found and st.session_state.apriori_rules is not None:
        rules = st.session_state.apriori_rules
        trans_df = st.session_state.trans_df
        product_df = st.session_state.product_df
        
        apriori = AprioriAlgorithm()
        
        st.divider()
        
        # Lựa chọn sản phẩm
        st.subheader("👈 Chọn Sản phẩm")
        
        # Lấy các product_id từ transaction
        product_ids = sorted(trans_df['PRODUCT_ID'].unique())
        selected_product_id = st.selectbox(
            "Sản phẩm gốc:",
            product_ids,
            format_func=lambda x: f"ID: {x}",
            key="product_select_recom"
        )
        
        # Tìm gợi ý
        recommendations = apriori.get_recommendations(selected_product_id, rules)
        
        if len(recommendations) == 0:
            st.info(f"Không có gợi ý cho sản phẩm ID: {selected_product_id}")
        else:
            st.subheader("💡 Sản phẩm Được Gợi ý")
            
            recom_df = pd.DataFrame({
                'PRODUCT_ID': recommendations
            })
            
            # Merge với product info
            if 'PRODUCT_ID' in product_df.columns:
                recom_df = recom_df.merge(product_df, 
                                         on='PRODUCT_ID', 
                                         how='left')
                
                # Tạo cột Tên Sản phẩm từ COMMODITY_DESC + SUB_COMMODITY_DESC
                if 'COMMODITY_DESC' in recom_df.columns:
                    recom_df['Tên Sản phẩm'] = recom_df.apply(
                        lambda row: f"{row['COMMODITY_DESC']} - {row['SUB_COMMODITY_DESC']}" 
                        if 'SUB_COMMODITY_DESC' in recom_df.columns 
                        else row['COMMODITY_DESC'],
                        axis=1
                    )
                
                # Chọn cột hiển thị
                display_cols = ['PRODUCT_ID', 'Tên Sản phẩm', 'BRAND', 'DEPARTMENT', 'CURR_SIZE_OF_PRODUCT']
                display_cols = [col for col in display_cols if col in recom_df.columns]
                
            st.dataframe(recom_df[display_cols], use_container_width=True)
            
            # Thống kê
            st.info(f"✅ Gợi ý {len(recommendations)} sản phẩm liên quan")
        
        st.divider()
        
        # Top Association Rules
        st.subheader("📊 Top 10 Luật Kết hợp (Highest Confidence)")
        top_rules = rules.nlargest(10, 'confidence')[
            ['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']
        ].reset_index(drop=True)
        top_rules.columns = ['Sản phẩm Trước', 'Sản phẩm Sau', 
                            'Support', 'Confidence', 'Lift']
        st.dataframe(top_rules, use_container_width=True)


def show_business_page():
    """Hiển thị trang Ứng Dụng Thực Tế chính."""
    st.title("📊 Ứng Dụng Thực Tế (Business Application)")
    st.write("""
    Bảng điều khiển tích hợp cho phân tích khách hàng và gợi ý sản phẩm.
    Sử dụng các mô hình Machine Learning để hỗ trợ quyết định kinh doanh.
    """)
    
    feature = st.sidebar.radio(
        "🎯 Chọn Tính năng:",
        [
            "� Phân tích Dữ liệu",
            "🔧 Tiền xử lý Dữ liệu",
            "👥 Phân khúc Khách hàng",
            "📢 Dự đoán Phản hồi Chiến dịch",
            "🛍️ Gợi ý Sản phẩm"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Quy trình:**\n1. 📊 Phân tích dữ liệu\n2. 🔧 Tiền xử lý\n3. 🤖 Phân tích ML")
    
    if "Phân tích" in feature and "Dữ" in feature:
        show_data_analysis()
    elif "Tiền xử lý" in feature:
        show_data_preprocessing()
    elif "Phân khúc" in feature:
        show_customer_segmentation()
    elif "Dự đoán" in feature:
        show_campaign_response_prediction()
    elif "Gợi ý" in feature:
        show_product_recommendation()
