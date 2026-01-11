"""
Giao diện Phòng Thí Nghiệm (Lab View) - Nơi trực quan hóa các thuật toán.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from src.data_layer import get_data_layer
from src.algorithms.apriori import AprioriAlgorithm
from src.algorithms.rough_set import RoughSetReduct
from src.algorithms.naive_bayes import NaiveBayesClassifier
from src.algorithms.decision_tree import DecisionTreeCART, DecisionTreeC45, DecisionTreeID3
from src.algorithms.bayesian_network import BayesianNetworkDAG
from src.algorithms.kmeans import KMeansClustering


def show_apriori_lab():
    """Hiển thị giao diện Lab cho Apriori Algorithm."""
    st.header("🎯 Thuật toán Apriori - Khai phá Luật Kết hợp")
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        st.info("""
        **Apriori** tìm các tập hợp sản phẩm (Frequent Itemsets) thường được mua cùng nhau
        và sinh ra các Luật Kết hợp (Association Rules).
        
        **Khái niệm chính:**
        - **Support**: Tỷ lệ % giao dịch chứa itemset
        - **Confidence**: Tỷ lệ % giao dịch có B nếu đã mua A
        - **Lift**: Độ mạnh của mối quan hệ (Lift > 1 = quan hệ dương)
        
        **Quá trình:**
        1. Tìm Frequent Itemsets (Support ≥ min_support)
        2. Sinh ra Association Rules từ Frequent Itemsets
        3. Lọc các Rules có Confidence ≥ min_confidence
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider(
            "Min Support (%)", 0.1, 10.0, 2.0, 0.1) / 100
    with col2:
        min_confidence = st.slider(
            "Min Confidence (%)", 10, 100, 50, 5) / 100
    
    if st.button("▶️ Chạy Apriori", key="apriori_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data(sample_size=30000)
            
            if len(trans_df) == 0:
                st.error("Không thể tải dữ liệu giao dịch.")
                return
            
            apriori = AprioriAlgorithm()
            itemsets, rules = apriori.run(trans_df, min_support, min_confidence)
            
            if len(itemsets) == 0:
                st.warning(f"Không tìm thấy Frequent Itemsets với min_support={min_support:.2%}")
                return
            
            st.success(f"✅ Tìm thấy {len(itemsets)} Frequent Itemsets")
            
            # Hiển thị Frequent Itemsets
            st.subheader("📊 Frequent Itemsets")
            itemsets_display = itemsets.copy()
            itemsets_display['itemsets'] = itemsets_display['itemsets'].apply(
                lambda x: ', '.join(str(i) for i in x)
            )
            st.dataframe(itemsets_display, use_container_width=True)
            
            # Hiển thị Association Rules
            if len(rules) > 0:
                st.subheader("🔗 Association Rules")
                rules_display = rules[['antecedent_str', 'consequent_str', 
                                       'support', 'confidence', 'lift']].copy()
                rules_display.columns = ['Sản phẩm Trước', 'Sản phẩm Sau', 
                                         'Support', 'Confidence', 'Lift']
                st.dataframe(rules_display.head(20), use_container_width=True)
                
                # Biểu đồ Support vs Confidence
                fig = go.Figure(data=go.Scatter(
                    x=rules['support'],
                    y=rules['confidence'],
                    mode='markers',
                    marker=dict(
                        size=rules['lift'] * 5,
                        color=rules['lift'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Lift")
                    ),
                    text=[f"{row['antecedent_str']} → {row['consequent_str']}" 
                          for _, row in rules.iterrows()],
                    hovertemplate="<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>"
                ))
                fig.update_layout(
                    title="📈 Support vs Confidence (kích thước = Lift)",
                    xaxis_title="Support",
                    yaxis_title="Confidence",
                    height=500,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không tìm thấy Association Rules với confidence này.")


def show_rough_set_lab():
    """Hiển thị giao diện Lab cho Rough Set."""
    st.header("🔍 Thuật toán Rough Set - Lựa chọn Đặc trưng")
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        st.info("""
        **Rough Set** sử dụng lý thuyết tập hợp để lựa chọn các đặc trưng quan trọng
        (Feature Selection) từ dữ liệu.
        
        **Khái niệm chính:**
        - **Reduct**: Tập hợp tối thiểu các đặc trưng vẫn giữ khả năng phân biệt
        - **Information Gain**: Độ giảm entropy khi sử dụng một đặc trưng
        - **Entropy**: Độ không chắc chắn / hỗn loạn của dữ liệu
        
        **Quá trình (Greedy):**
        1. Tính Information Gain cho từng đặc trưng
        2. Chọn đặc trưng có Gain cao nhất
        3. Lặp lại cho đến khi đạt số lượng max hoặc gain <= 0
        """)
    
    max_features = st.slider("Số đặc trưng tối đa", 1, 10, 5)
    
    if st.button("▶️ Chạy Rough Set", key="rough_set_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            merged = data_layer.get_merged_dataset(sample_size=10000)
            
            if len(merged) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            # Chọn các cột liên quan
            demo_cols = ['AGE_DESC', 'INCOME_DESC', 'MARITAL_STATUS_CODE', 
                        'HOMEOWNER_DESC']
            df_selected = merged[demo_cols].copy()
            
            # Tạo cột mục tiêu (chi tiêu cao/thấp)
            median_spending = merged['SALES_VALUE'].median()
            df_selected['CHI_TIEU_CAO'] = (merged['SALES_VALUE'] > median_spending).astype(int)
            
            rough_set = RoughSetReduct()
            result = rough_set.run(df_selected, target='CHI_TIEU_CAO', 
                                  max_features=max_features)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Đã lựa chọn {result['num_features_selected']} đặc trưng")
            
            # Hiển thị kết quả
            st.subheader("🎯 Reduct (Đặc trưng được chọn)")
            st.write(f"**Đặc trưng:** {', '.join(result['reduct'])}")
            
            # Biểu đồ Information Gain
            if result['importance_scores']:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(result['importance_scores'].keys()),
                        y=list(result['importance_scores'].values()),
                        marker=dict(color='steelblue')
                    )
                ])
                fig.update_layout(
                    title="📊 Information Gain của từng Đặc trưng",
                    xaxis_title="Đặc trưng",
                    yaxis_title="Information Gain",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


def show_naive_bayes_lab():
    """Hiển thị giao diện Lab cho Naïve Bayes."""
    st.header("🤖 Thuật toán Naïve Bayes - Phân lớp Xác suất")
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        st.info("""
        **Naïve Bayes** sử dụng Định lý Bayes để dự đoán xác suất một mẫu thuộc về mỗi lớp.
        
        **Công thức Bayes:**
        P(Lớp|Đặc trưng) = P(Đặc trưng|Lớp) × P(Lớp) / P(Đặc trưng)
        
        **Giả định Naïve:** Tất cả đặc trưng độc lập với nhau (không có mối liên hệ).
        
        **Laplace Smoothing:** Thêm 1 vào tử số và mẫu số để xử lý Zero Probability Problem.
        Công thức: P(x|y) = (count(x,y) + 1) / (count(y) + num_classes)
        """)
    
    use_laplace = st.checkbox("✓ Sử dụng Laplace Smoothing", value=True)
    
    if st.button("▶️ Chạy Naïve Bayes", key="naive_bayes_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            merged = data_layer.get_merged_dataset(sample_size=10000)
            
            if len(merged) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            clf = NaiveBayesClassifier(use_laplace_smoothing=use_laplace)
            result = clf.train(merged)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Huấn luyện thành công (Accuracy: {result['accuracy']:.2%})")
            
            # Hiển thị kết quả
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Độ chính xác (Accuracy)", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{result['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{result['recall']:.2%}")
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = result['confusion_matrix']
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Âm tính', 'Dương tính'],
                y=['Âm tính', 'Dương tính'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Dự đoán",
                yaxis_title="Thực tế",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def show_decision_tree_lab():
    """Hiển thị giao diện Lab cho Decision Tree (CART, C4.5, ID3)."""
    st.header("🌳 Cây Quyết định (Decision Tree)")
    
    tree_type = st.radio("Chọn loại cây:", 
                         ["CART (Gini Impurity)", "C4.5 (Information Gain)", 
                          "ID3 (Entropy)"])
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        if tree_type == "CART (Gini Impurity)":
            st.info("""
            **CART** sử dụng **Gini Impurity** để tìm điểm phân chia tốt nhất.
            
            **Gini Impurity:** G = 1 - Σ(p_i)²
            - Gini = 0: Nút thuần chủng (tất cả một lớp)
            - Gini = 0.5: Nút hỗn hợp (chia đều giữa các lớp)
            """)
        elif tree_type == "C4.5 (Information Gain)":
            st.info("""
            **C4.5 (Quinlan)** sử dụng **Information Gain Ratio**.
            
            **Information Gain:** IG = H(Parent) - Σ(|Child|/|Parent|) × H(Child)
            **Entropy:** H = -Σ(p_i × log₂(p_i))
            - Entropy = 0: Nút thuần chủng
            - Entropy = 1: Nút hỗn hợp (2 lớp)
            """)
        else:
            st.info("""
            **ID3** là phiên bản đơn giản của C4.5, cũng dùng **Entropy**.
            Khác biệt: ID3 không có pruning, dễ Overfitting trên dữ liệu nhỏ.
            """)
    
    max_depth = st.slider("Độ sâu tối đa (max_depth)", 3, 15, 5)
    min_samples = st.slider("Min samples để tách nút", 2, 20, 10)
    
    if st.button("▶️ Chạy Decision Tree", key="dt_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            merged = data_layer.get_merged_dataset(sample_size=10000)
            
            if len(merged) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            if tree_type == "CART (Gini Impurity)":
                model = DecisionTreeCART(max_depth=max_depth, 
                                        min_samples_split=min_samples)
            elif tree_type == "C4.5 (Information Gain)":
                model = DecisionTreeC45(max_depth=max_depth,
                                       min_samples_split=min_samples)
            else:
                model = DecisionTreeID3(max_depth=max_depth,
                                       min_samples_split=min_samples)
            
            result = model.train(merged)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Huấn luyện thành công (Accuracy: {result['accuracy']:.2%})")
            
            # Thông tin cây
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Độ chính xác", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Số nút", result['num_nodes'])
            with col3:
                st.metric("Độ sâu", result['max_depth'])
            
            # Feature Importance
            st.subheader("🎯 Feature Importance (Tầm quan trọng Đặc trưng)")
            features_df = pd.DataFrame(list(result['feature_importance'].items()),
                                      columns=['Đặc trưng', 'Tầm quan trọng'])
            features_df = features_df.sort_values('Tầm quan trọng', ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(x=features_df['Tầm quan trọng'],
                      y=features_df['Đặc trưng'],
                      orientation='h',
                      marker=dict(color='teal'))
            ])
            fig.update_layout(
                title="Tầm quan trọng của từng Đặc trưng",
                xaxis_title="Tầm quan trọng",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def show_bayesian_network_lab():
    """Hiển thị giao diện Lab cho Bayesian Network."""
    st.header("🕸️ Bayesian Network - Mô hình Xác suất Đồ thị")
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        st.info("""
        **Bayesian Network** là một đồ thị có hướng (DAG) biểu diễn các mối quan hệ
        xác suất giữa các biến.
        
        **Cấu trúc DAG:** Tuổi (AGE) → Thu nhập (INCOME) → Sở hữu nhà (HOMEOWNER)
        
        **Ý nghĩa:**
        - Tuổi ảnh hưởng đến Thu nhập
        - Thu nhập ảnh hưởng đến Sở hữu nhà
        
        **Suy diễn (Inference):**
        Cho trước giá trị tuổi, tính xác suất P(INCOME|AGE) và P(HOMEOWNER|AGE).
        """)
    
    if st.button("▶️ Chạy Bayesian Network", key="bn_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            merged = data_layer.get_merged_dataset(sample_size=10000)
            
            if len(merged) == 0:
                st.error("Không thể tải dữ liệu.")
                return
            
            bn = BayesianNetworkDAG()
            result = bn.fit(merged)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Huấn luyện thành công ({result['num_samples']} mẫu)")
            
            # Hiển thị cấu trúc DAG
            st.subheader("📊 Cấu trúc DAG (Directed Acyclic Graph)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Nút (Nodes):**")
                st.write(", ".join(result['structure']['nodes']))
            
            with col2:
                st.write("**Cạnh (Edges):**")
                edges_str = " → ".join([f"{e[0]}→{e[1]}" for e in result['structure']['edges']])
                st.write(edges_str)
            
            # Vẽ DAG với bố cục tốt hơn
            fig = go.Figure()
            
            # Tọa độ nút với bố cục ngang
            node_positions = {
                'AGE': (0, 2),
                'INCOME': (2, 2),
                'HOMEOWNER': (4, 2)
            }
            
            # Vẽ cạnh (edges) với mũi tên
            edges = [('AGE', 'INCOME'), ('INCOME', 'HOMEOWNER')]
            for source, target in edges:
                x0, y0 = node_positions[source]
                x1, y1 = node_positions[target]
                
                # Vẽ đường
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 255, 0.5)', width=3),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Thêm mũi tên (tam giác nhỏ ở cuối)
                fig.add_annotation(
                    x=x1, y=y1,
                    ax=x0, ay=y0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=3,
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor='rgba(100, 100, 255, 0.7)',
                    showarrow=True
                )
            
            # Vẽ nút (nodes) với màu sắc khác nhau
            node_colors = {'AGE': 'lightcoral', 'INCOME': 'lightyellow', 'HOMEOWNER': 'lightgreen'}
            node_x = [node_positions[node][0] for node in ['AGE', 'INCOME', 'HOMEOWNER']]
            node_y = [node_positions[node][1] for node in ['AGE', 'INCOME', 'HOMEOWNER']]
            node_colors_list = [node_colors[node] for node in ['AGE', 'INCOME', 'HOMEOWNER']]
            
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=60,
                    color=node_colors_list,
                    line=dict(color='darkblue', width=3)
                ),
                text=['AGE', 'INCOME', 'HOMEOWNER'],
                textposition='middle center',
                textfont=dict(size=14, color='black', family='Arial Black'),
                hoverinfo='text',
                showlegend=False
            ))
            
            fig.update_layout(
                title="🕸️ Bayesian Network DAG: Age → Income → Homeowner",
                showlegend=False,
                hovermode='closest',
                height=400,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 3]),
                plot_bgcolor='rgba(240, 240, 250, 0.5)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị Conditional Probability Tables
            st.subheader("📋 Conditional Probability Distribution (CPD)")
            
            tab1, tab2, tab3 = st.tabs(["P(AGE)", "P(INCOME|AGE)", "P(HOMEOWNER|INCOME)"])
            
            with tab1:
                st.write("**Prior Probability - P(AGE):**")
                if 'cpd_age' in result:
                    cpd_age_df = pd.DataFrame(list(result['cpd_age'].items()), 
                                              columns=['Age Category', 'Probability'])
                    st.dataframe(cpd_age_df, use_container_width=True)
            
            with tab2:
                st.write("**Conditional Probability - P(INCOME|AGE):**")
                if 'cpd_income_given_age' in result:
                    cpd_income = result['cpd_income_given_age']
                    income_data = []
                    for age_key, income_dict in cpd_income.items():
                        for income_key, prob in income_dict.items():
                            income_data.append({
                                'Age Category': age_key,
                                'Income Category': income_key,
                                'Probability': round(prob, 4)
                            })
                    if income_data:
                        cpd_income_df = pd.DataFrame(income_data)
                        st.dataframe(cpd_income_df, use_container_width=True)
                    else:
                        st.info("Không có dữ liệu CPD Income|Age")
            
            with tab3:
                st.write("**Conditional Probability - P(HOMEOWNER|INCOME):**")
                if 'cpd_homeowner_given_income' in result:
                    cpd_homeowner = result['cpd_homeowner_given_income']
                    homeowner_data = []
                    for income_key, homeowner_dict in cpd_homeowner.items():
                        for homeowner_key, prob in homeowner_dict.items():
                            homeowner_data.append({
                                'Income Category': income_key,
                                'Homeowner Status': homeowner_key,
                                'Probability': round(prob, 4)
                            })
                    if homeowner_data:
                        cpd_homeowner_df = pd.DataFrame(homeowner_data)
                        st.dataframe(cpd_homeowner_df, use_container_width=True)
                    else:
                        st.info("Không có dữ liệu CPD Homeowner|Income")


def show_kmeans_lab():
    """Hiển thị giao diện Lab cho k-Means Clustering."""
    st.header("🎨 k-Means Clustering - Phân nhóm RFM")
    
    with st.expander("📖 Nguyên lý hoạt động", expanded=False):
        st.info("""
        **k-Means** là thuật toán phân nhóm không giám sát chia dữ liệu thành k cụm.
        
        **RFM (Recency, Frequency, Monetary):**
        - **Recency:** Ngày kể từ lần mua cuối (càng gần = càng tốt)
        - **Frequency:** Số lần mua (càng nhiều = càng tốt)
        - **Monetary:** Tổng chi tiêu (càng cao = càng tốt)
        
        **Quá trình:**
        1. Khởi tạo k tâm cụm ngẫu nhiên
        2. Gán mỗi điểm đến cụm gần nhất
        3. Cập nhật tâm cụm (trung bình các điểm)
        4. Lặp lại 2-3 cho đến hội tụ
        
        **Chỉ số đánh giá:**
        - **Silhouette Score:** (-1, 1), cao hơn = tốt hơn
        - **Davies-Bouldin Index:** < 1 = tốt
        """)
    
    n_clusters = st.slider("Số cụm (k)", 2, 10, 3)
    
    if st.button("▶️ Chạy k-Means", key="kmeans_run"):
        with st.spinner("Đang xử lý..."):
            data_layer = get_data_layer()
            trans_df = data_layer.load_transaction_data(sample_size=30000)
            
            if len(trans_df) == 0:
                st.error("Không thể tải dữ liệu giao dịch.")
                return
            
            kmeans = KMeansClustering(n_clusters=n_clusters)
            result = kmeans.fit(trans_df)
            
            if 'error' in result:
                st.error(result['error'])
                return
            
            st.success(f"✅ Phân nhóm thành công ({result['n_samples']} khách hàng)")
            
            # Chỉ số đánh giá
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
            with col2:
                st.metric("Davies-Bouldin Index", f"{result['davies_bouldin_score']:.3f}")
            with col3:
                st.metric("Inertia", f"{result['inertia']:.0f}")
            
            # Thống kê cụm
            rfm_data = result['rfm_data']
            cluster_stats = kmeans.get_cluster_statistics(rfm_data)
            
            st.subheader("📊 Thống kê Cụm")
            for cluster_name, stats in cluster_stats.items():
                st.write(f"**{cluster_name}:**")
                col_info = st.columns(4)
                col_info[0].metric("Khách hàng", f"{stats['Số khách hàng']:.0f}")
                col_info[1].metric("Recency (ngày)", f"{stats['Recency trung bình']:.1f}")
                col_info[2].metric("Frequency", f"{stats['Frequency trung bình']:.1f}")
                col_info[3].metric("Monetary", f"{stats['Monetary trung bình']:.0f}")
            
            # Biểu đồ 3D RFM
            st.subheader("🎯 Phân bố 3D RFM")
            fig = px.scatter_3d(rfm_data,
                               x='Recency', y='Frequency', z='Monetary',
                               color='Cluster',
                               title='Phân bố Khách hàng theo RFM',
                               labels={'Cluster': 'Cụm'})
            st.plotly_chart(fig, use_container_width=True)


def show_lab_page():
    """Hiển thị trang Phòng Thí Nghiệm chính."""
    st.title("🧪 Phòng Thí Nghiệm (Academic Lab)")
    st.write("Nơi trực quan hóa và tinh chỉnh 9 thuật toán Khoa học Dữ liệu")
    
    algorithm = st.sidebar.radio(
        "🎯 Chọn Thuật toán:",
        [
            "1️⃣ Apriori",
            "2️⃣ Rough Set",
            "3️⃣ Naïve Bayes",
            "4️⃣ Decision Tree",
            "5️⃣ Bayesian Network",
            "6️⃣ k-Means Clustering"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Ghi chú:** Các mô hình sử dụng lấy mẫu dữ liệu để tối ưu hiệu năng")
    
    if "Apriori" in algorithm:
        show_apriori_lab()
    elif "Rough Set" in algorithm:
        show_rough_set_lab()
    elif "Naïve Bayes" in algorithm:
        show_naive_bayes_lab()
    elif "Decision Tree" in algorithm:
        show_decision_tree_lab()
    elif "Bayesian Network" in algorithm:
        show_bayesian_network_lab()
    elif "k-Means" in algorithm:
        show_kmeans_lab()
