# Hệ thống Bán lẻ Thông minh (Retail Smart System)

### Tạo Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Cài Dependencies
```bash
pip install -r requirements.txt
```

### Chạy Ứng dụng
```bash
streamlit run app.py
```

Truy cập: `http://localhost:8501`

---

## Cấu trúc Dự án

```
IE403.F11.CN2.CNTT/
├── app.py                      # Entry point
├── requirements.txt            # Dependencies (+ kagglehub)
├── .gitignore                  # Ignore data/, secrets.toml
├── README.md                   # This file
│
├── data/                       # CSV files (git ignored)
│   ├── transaction_data.csv
│   ├── product.csv
│   ├── hh_demographic.csv
│   └── ...
│
├── src/
│   ├── config.py              # Configuration
│   ├── data_layer.py          # Data access + Kaggle download
│   ├── data_preprocessing.py  # Data analysis & cleaning
│   │
│   ├── algorithms/
│   │   ├── apriori.py
│   │   ├── rough_set.py
│   │   ├── naive_bayes.py
│   │   ├── decision_tree.py
│   │   ├── bayesian_network.py
│   │   └── kmeans.py
│   │
│   └── ui/
│       ├── lab_view.py        # Lab mode (9 algorithms)
│       └── business_view.py   # Business mode (5 features)
│
└── .streamlit/
    ├── config.toml            # Streamlit settings
    └── secrets.toml.example   # Example secrets (copy → secrets.toml)
```

---

## Cách Hoạt Động (Local vs Cloud)

### Local (`data/` Tồn Tại)
```
app.py chạy
    ↓
data_layer.py kiểm tra `data/` folder
    ↓
✅ Tìm thấy CSV files
    ↓
Load từ local (nhanh)
```

### Streamlit Cloud (Không có `data/`)
```
app.py chạy
    ↓
data_layer.py kiểm tra `data/` folder
    ↓
❌ Không tìm thấy
    ↓
Gọi KaggleDownloader
    ↓
Lấy kaggle_username, kaggle_key từ secrets.toml
    ↓
Download từ Kaggle: vjchoudhary7/customer-segmentation-tutorial-in-python
    ↓
Cache dữ liệu (lần sau dùng ngay)
```

---


## Chế Độ Phòng Thí Nghiệm (Lab)

6 thuật toán tương tác:

| Thuật toán | Tác Dụng | Input | Output |
|-----------|---------|-------|--------|
| **Apriori** | Khai phá quy tắc kết hợp | Min Support, Confidence | Association Rules |
| **Rough Set** | Lựa chọn đặc trưng | Max Features | Feature Importance |
| **Naive Bayes** | Phân lớp xác suất | Laplace toggle | Confusion Matrix |
| **Decision Tree** | 3 biến thể (CART/C4.5/ID3) | Algorithm choice | Feature Importance |
| **Bayesian Network** | Mô hình xác suất đồ thị | DAG structure | CPD Tables |
| **k-Means** | RFM clustering | k value | Silhouette Score |

---

## Chế Độ Ứng Dụng Thực Tế (Business)

5 features cho business:

1. **Phân Tích Dữ Liệu** - Dataset statistics
2. **Tiền Xử Lý** - Data cleaning & validation
3. **Phân Khúc Khách** - RFM clustering
4. **Dự Đoán Chiến Dịch** - Campaign response prediction
5. **Gợi Ý Sản Phẩm** - Association rules recommendations

---

## Cấu Hình Tùy Chỉnh

### File: `src/config.py`
```python
# Apriori parameters
APRIORI_CONFIG = {
    'min_support': 0.001,
    'min_confidence': 0.3,
}

# k-Means parameters
KMEANS_CONFIG = {
    'max_k': 10,
    'random_state': 42,
}
```

---