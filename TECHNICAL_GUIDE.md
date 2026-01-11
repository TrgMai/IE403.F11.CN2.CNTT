# HỆ THỐNG BÁN LẺ THÔNG MINH - TECHNICAL GUIDE

## 1. KIẾN TRÚC CHUNG

### Mô hình: Modular Monolith

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APP                        │
│                    (app.py - Main Entry)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │  LAB (lab_view)  │          │  BUSINESS (...)  │         │
│  │                  │          │                  │         │
│  │  9 Thuật toán    │          │  3 Tính năng     │         │
│  └────────┬─────────┘          └────────┬─────────┘         │
│           │                             │                   │
├───────────┴─────────────────────────────┴───────────────────┤
│                   ALGORITHMS LAYER                          │
│                   (src/algorithms/)                         │
│                                                             │
│  ┌──────────┬─────────┬──────────┬──────────┬────────┐      │
│  │ Apriori  │Rough Set│Naive Bay│Decision T│Bayesian │      │
│  └──────────┴─────────┴──────────┴──────────┴────────┘      │
│                                                             │
│  ┌──────────┐                                               │
│  │ k-Means  │                                               │
│  └──────────┘                                               │
├─────────────────────────────────────────────────────────────┤
│                   DATA LAYER                                │
│           (src/data_layer.py - Singleton)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              DATA CACHING (Streamlit @cache)                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   CSV FILES (data/)                         │
│                                                             │
│  transaction_data | product | demographic | campaign        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Alg tính: 

1. **Lớp UI (Presentation):** Streamlit views
   - `lab_view.py`: Giao diện 9 thuật toán
   - `business_view.py`: Giao diện 3 tính năng kinh doanh

2. **Lớp Thuật toán (Business Logic):**
   - `algorithms/apriori.py`
   - `algorithms/rough_set.py`
   - `algorithms/naive_bayes.py`
   - `algorithms/decision_tree.py` (CART, C4.5, ID3)
   - `algorithms/bayesian_network.py`
   - `algorithms/kmeans.py`

3. **Lớp Dữ liệu (Data Access):**
   - `data_layer.py`: Singleton pattern
   - Caching với @st.cache_data
   - Data sampling & merge

4. **Entry Point:**
   - `app.py`: Navigation & routing

---

## 2. THÀNH PHẦN CHI TIẾT

### A. DATA_LAYER.PY (Singleton Pattern)

**Mục đích:** Quản lý tải dữ liệu từ CSV, áp dụng caching, và sampling.

**Class chính:**
```python
class DataLayerSingleton:
    - load_transaction_data(sample_size=50000)
    - load_product_data()
    - load_demographic_data()
    - load_campaign_data()
    - load_coupon_data()
    - load_coupon_redemption_data()
    - get_merged_dataset(sample_size=50000)
```

**Caching Strategy:**
```python
@st.cache_data  # Cache trên bộ nhớ Streamlit
def load_transaction_data(_self, sample_size=50000):
    # Cached - chỉ tải một lần
    return df
```

**Sampling:** Giới hạn 50,000 dòng mặc định để tối ưu hiệu năng

---

### B. APRIORI.PY - Association Rule Mining

**Khái niệm:**
- **Itemset:** Tập hợp sản phẩm mua trong 1 giao dịch
- **Frequent Itemset:** Itemset có support ≥ min_support
- **Association Rule:** A → B (nếu mua A, khả năng mua B)

**Metrics:**
- **Support(A→B):** P(A ∩ B) = số {A,B} / tổng giao dịch
- **Confidence(A→B):** P(B|A) = số {A,B} / số {A}
- **Lift(A→B):** Confidence(A→B) / Support(B)

**Methods:**
```python
class AprioriAlgorithm:
    run(transaction_df, min_support, min_confidence)
        → Trả về (itemsets, rules)
    
    get_recommendations(product_id, rules)
        → Trả về danh sách sản phẩm gợi ý
```

---

### C. ROUGH_SET.PY - Feature Selection

**Khái niệm:**
- **Reduct:** Tập hợp tối thiểu đặc trưng vẫn giữ khả năng phân biệt
- **Information Gain:** IG = H(Parent) - H(Parent|Feature)
- **Entropy:** H = -Σ(p_i × log₂(p_i))

**Thuật toán (Greedy):**
1. Tính Information Gain cho tất cả đặc trưng
2. Chọn đặc trưng có IG cao nhất
3. Loại bỏ đặc trưng đó
4. Lặp lại cho đến khi đạt max_features hoặc IG ≤ 0

**Methods:**
```python
class RoughSetReduct:
    run(df, target, max_features)
        → Trả về reduct & importance_scores
    
    calculate_information_gain(df, feature, target)
        → Tính IG của một đặc trưng
```

---

### D. NAIVE_BAYES.PY - Probabilistic Classification

**Công thức:**
P(Lớp|Đặc trưng) = P(Đặc trưng|Lớp) × P(Lớp) / P(Đặc trưng)

**Giả định Naïve:** Tất cả đặc trưng độc lập nhau

**Laplace Smoothing:**
P(x|y) = (count(x,y) + 1) / (count(y) + num_classes)
- Xử lý Zero Probability Problem
- Có thể bật/tắt qua checkbox trong UI

**Methods:**
```python
class NaiveBayesClassifier:
    train(df, target_column)
        → Trả về accuracy, precision, recall, confusion_matrix
    
    predict(X) → Dự đoán nhãn
    get_probabilities(X) → Lấy xác suất từng lớp
```

---

### E. DECISION_TREE.PY - Tree-Based Classification

**3 Biến thể:**

#### 1. CART (Classification And Regression Trees)
- **Tiêu chí chia:** Gini Impurity = 1 - Σ(p_i²)
- **Gini = 0:** Nút thuần chủng
- **Gini = 0.5:** Nút hỗn hợp (2 lớp)

#### 2. C4.5 (Quinlan Algorithm)
- **Tiêu chí chia:** Information Gain Ratio
- IG = H(Parent) - Σ(|Child|/|Parent|) × H(Child)
- **Entropy:** H = -Σ(p_i × log₂(p_i))
- Cải tiến ID3 với pruning

#### 3. ID3 (Iterative Dichotomiser 3)
- **Tiêu chí chia:** Information Gain (Entropy)
- Phiên bản đơn giản, không pruning
- Dễ Overfitting trên dữ liệu nhỏ

**Methods:**
```python
class DecisionTreeCART/C45/ID3:
    train(df, target_column)
        → Trả về accuracy, feature_importance, confusion_matrix
```

---

### F. BAYESIAN_NETWORK.PY - Probabilistic Graphical Model

**Cấu trúc DAG (Directed Acyclic Graph):**
```
Age → Income → Homeowner
```

**CPD (Conditional Probability Distribution):**
- P(Age): Prior probability
- P(Income|Age): Conditional probability
- P(Homeowner|Income): Conditional probability

**Suy diễn:**
- Input: Giá trị Age
- Output: P(Income|Age), P(Homeowner|Age)

**Methods:**
```python
class BayesianNetworkDAG:
    fit(df) → Huấn luyện mô hình
    predict_inference(age_value) → Suy diễn xác suất
    get_dag_structure() → Lấy cấu trúc DAG
```

---

### G. KMEANS.PY - Unsupervised Clustering

**RFM Analysis:**
- **Recency (R):** Ngày kể từ lần mua cuối (ngắn = tốt)
- **Frequency (F):** Số lần mua (cao = tốt)
- **Monetary (M):** Tổng chi tiêu (cao = tốt)

**Thuật toán:**
1. Khởi tạo k tâm cụm ngẫu nhiên
2. Gán mỗi điểm → cụm gần nhất
3. Cập nhật tâm cụm (trung bình)
4. Lặp lại 2-3 cho đến hội tụ

**Đánh giá:**
- **Silhouette Score:** (-1, 1), cao hơn = tốt
- **Davies-Bouldin Index:** < 1 = tốt
- **Inertia:** Tổng khoảng cách bình phương

**Methods:**
```python
class KMeansClustering:
    fit(transaction_df)
        → Trả về cluster_labels, statistics
    
    calculate_rfm(transaction_df)
        → Tính chỉ số RFM
    
    predict(new_rfm)
        → Dự đoán cụm cho dữ liệu mới
```

---

## 3. UI LAYER

### LAB_VIEW.PY - Phòng Thí Nghiệm

**6 Hàm chính (mỗi hàm = 1 tab):**

1. `show_apriori_lab()`
   - Thanh trượt: min_support, min_confidence
   - Output: Itemsets table, Rules table, Support vs Confidence plot

2. `show_rough_set_lab()`
   - Input: max_features
   - Output: Reduct, Information Gain chart

3. `show_naive_bayes_lab()`
   - Checkbox: Laplace Smoothing
   - Output: Confusion Matrix, Accuracy, Precision, Recall

4. `show_decision_tree_lab()`
   - Radio: CART vs C4.5 vs ID3
   - Sliders: max_depth, min_samples_split
   - Output: Feature Importance chart, Confusion Matrix

5. `show_bayesian_network_lab()`
   - Output: DAG visualization, CPD tables

6. `show_kmeans_lab()`
   - Slider: n_clusters
   - Output: Silhouette Score, 3D RFM scatter plot

---

### BUSINESS_VIEW.PY - Ứng Dụng Thực Tế

**3 Hàm chính:**

1. `show_customer_segmentation()`
   - k-Means trên RFM
   - Chiến lược kinh doanh cho mỗi segment
   - Pie chart & 3D scatter

2. `show_campaign_response_prediction()`
   - Decision Tree dự đoán phản hồi
   - Feature Importance
   - Confusion Matrix

3. `show_product_recommendation()`
   - Apriori tìm Association Rules
   - User chọn product_id
   - Hiển thị gợi ý

---

## 4. FLOW DIAGRAM

### Flow Khi User Chạy Apriori:

```
User nhấp "Chạy Apriori"
    ↓
Streamlit trigger callback
    ↓
data_layer.load_transaction_data(30000)
    ├─ Kiểm tra cache (@st.cache_data)
    ├─ Nếu cached → Return cached data
    └─ Nếu không → Load CSV, apply sampling, cache
    ↓
AprioriAlgorithm.run(trans_df, min_support, min_confidence)
    ├─ prepare_transaction_data() → Nhóm sản phẩm theo giỏ hàng
    ├─ TransactionEncoder() → One-hot encoding
    ├─ apriori() → Tìm Frequent Itemsets
    ├─ association_rules() → Sinh Association Rules
    └─ Sắp xếp theo Confidence
    ↓
Streamlit hiển thị:
    ├─ Success message
    ├─ Frequent Itemsets table
    ├─ Association Rules table
    └─ Scatter plot (Support vs Confidence)
```

---

## 5. CACHING STRATEGY

### Streamlit @st.cache_data

```python
@st.cache_data
def load_transaction_data(_self, sample_size=50000):
    # Cache key: tên hàm + tham số
    # Hết cache nếu: user tắt ứng dụng hoặc file code thay đổi
    
    df = pd.read_csv(...)
    if len(df) > sample_size:
        df = df.head(sample_size)
    return df
```

**Lợi ích:**
- Tải CSV 1 lần duy nhất
- Lần sau chạy instant
- Tiết kiệm RAM & disk I/O

**Thời gian:**
- Lần 1: 2-3 giây (load + cache)
- Lần 2+: < 100ms (cached)

---