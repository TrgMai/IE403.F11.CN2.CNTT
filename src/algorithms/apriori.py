"""
Thuật toán Apriori - Khai phá luật kết hợp (Association Rule Mining).
Tìm tập hợp sản phẩm thường được mua cùng nhau.
Tối ưu: Xử lý dữ liệu hiệu quả hơn, giảm memory usage.
"""
import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Tuple, List
from src.config import APRIORI_CONFIG
import gc

class AprioriAlgorithm:
    """
    Lớp triển khai thuật toán Apriori với tối ưu memory.
    
    Thuật toán này tìm Frequent Itemsets (tập vật phẩm thường xuyên xuất hiện)
    và Association Rules (luật kết hợp) từ dữ liệu giao dịch.
    """
    
    def __init__(self):
        """Khởi tạo Apriori Algorithm."""
        self.frequent_itemsets = None
        self.rules = None
    
    def prepare_transaction_data(self, transaction_df: pd.DataFrame) -> List[List]:
        """
        Chuẩn bị dữ liệu giao dịch từ định dạng DataFrame thành danh sách giao dịch.
        
        Tham số:
        - transaction_df (pd.DataFrame): DataFrame chứa cột BASKET_ID và PRODUCT_ID
        
        Trả về:
        - List[List]: Danh sách các giao dịch, mỗi giao dịch là danh sách PRODUCT_ID
        """
        # Nhóm các sản phẩm theo BASKET_ID (mỗi giỏ hàng)
        transactions = transaction_df.groupby('BASKET_ID')['PRODUCT_ID'].apply(list).values.tolist()
        return transactions
    
    def run(self, transaction_df: pd.DataFrame, 
            min_support: float = 0.01, 
            min_confidence: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chạy thuật toán Apriori với tối ưu memory.
        
        Tham số:
        - transaction_df (pd.DataFrame): Dữ liệu giao dịch
        - min_support (float): Ngưỡng hỗ trợ tối thiểu (0-1). Mặc định 0.01 (1%).
        - min_confidence (float): Ngưỡng độ tin cậy tối thiểu (0-1). Mặc định 0.5.
        
        Trả về:
        - Tuple[pd.DataFrame, pd.DataFrame]: 
            - frequent_itemsets: Bảng Frequent Itemsets
            - rules: Bảng Association Rules
        """
        try:
            # Chuẩn bị dữ liệu giao dịch
            transactions = self.prepare_transaction_data(transaction_df)
            
            if not transactions:
                st.warning("Không có dữ liệu giao dịch để xử lý.")
                return pd.DataFrame(), pd.DataFrame()
            
            st.info(f"📊 Đang xử lý {len(transactions):,} giao dịch...")
            
            # Chuyển đổi danh sách giao dịch thành one-hot encoded DataFrame
            # Sử dụng sparse matrix để tiết kiệm memory
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Xóa transactions từ memory
            del transactions
            gc.collect()
            
            st.info(f"✅ Đã load {df_encoded.shape[0]:,} giao dịch, {df_encoded.shape[1]} sản phẩm")
            st.info(f"⚙️ Tìm Frequent Itemsets với min_support={min_support:.2%}...")
            
            # Tìm Frequent Itemsets
            self.frequent_itemsets = apriori(df_encoded, 
                                              min_support=min_support, 
                                              use_colnames=True,
                                              max_len=3)  # Giới hạn độ dài itemset
            
            # Xóa df_encoded từ memory
            del df_encoded
            gc.collect()
            
            if len(self.frequent_itemsets) == 0:
                st.warning(f"⚠️ Không tìm thấy Frequent Itemsets với min_support={min_support:.2%}")
                st.info("💡 Gợi ý: Hãy giảm min_support hoặc tăng sample_size")
                return self.frequent_itemsets, pd.DataFrame()
            
            st.success(f"✅ Tìm thấy {len(self.frequent_itemsets)} Frequent Itemsets")
            st.info(f"⚙️ Tìm Association Rules với min_confidence={min_confidence:.1%}...")
            
            # Tìm Association Rules
            if len(self.frequent_itemsets) > 1:
                self.rules = association_rules(self.frequent_itemsets, 
                                                metric='confidence', 
                                                min_threshold=min_confidence)
            else:
                self.rules = pd.DataFrame()
            
            # Thêm cột hữu ích khác
            if len(self.rules) > 0:
                self.rules['antecedent_str'] = self.rules['antecedents'].apply(
                    lambda x: ', '.join(str(i) for i in list(x))
                )
                self.rules['consequent_str'] = self.rules['consequents'].apply(
                    lambda x: ', '.join(str(i) for i in list(x))
                )
                self.rules = self.rules.sort_values('confidence', ascending=False)
                st.success(f"✅ Tìm thấy {len(self.rules)} Association Rules")
            else:
                st.warning("⚠️ Không tìm thấy Association Rules")
            
            return self.frequent_itemsets, self.rules
        
        except MemoryError as e:
            st.error(f"❌ Lỗi Memory: Dữ liệu quá lớn!")
            st.warning("""
            **Giải pháp:**
            - ⬇️ Giảm Sample Size trong Cấu hình (⚙️ Config)
            - ⬆️ Tăng Min Support để giảm itemsets
            - 💾 Đóng các tab/ứng dụng khác để giải phóng RAM
            """)
            return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            st.error(f"❌ Lỗi khi chạy Apriori: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_recommendations(self, product_id: int, rules_df: pd.DataFrame) -> List[int]:
        """
        Gợi ý các sản phẩm liên quan dựa trên luật kết hợp.
        
        Tham số:
        - product_id (int): ID sản phẩm cần tìm gợi ý
        - rules_df (pd.DataFrame): Bảng luật kết hợp
        
        Trả về:
        - List[int]: Danh sách ID sản phẩm được gợi ý
        """
        recommendations = []
        
        for _, row in rules_df.iterrows():
            # antecedents và consequents là frozensets
            antecedents_set = row['antecedents']
            consequents_set = row['consequents']
            
            # Kiểm tra nếu product_id nằm trong antecedents
            # Chuyển antecedents_set thành list để kiểm tra
            if product_id in antecedents_set:
                # Thêm các sản phẩm trong consequents
                recommendations.extend(list(consequents_set))
        
        # Nếu không tìm thấy product_id trong antecedents, 
        # thử tìm trong consequents (sản phẩm được mua cùng)
        if not recommendations:
            for _, row in rules_df.iterrows():
                consequents_set = row['consequents']
                if product_id in consequents_set:
                    # Thêm các sản phẩm trong antecedents
                    recommendations.extend(list(row['antecedents']))
        
        return list(set(recommendations))  # Loại bỏ trùng lặp

    
    def get_recommendations(self, product_id: int, rules_df: pd.DataFrame) -> List[int]:
        """
        Gợi ý các sản phẩm liên quan dựa trên luật kết hợp.
        
        Tham số:
        - product_id (int): ID sản phẩm cần tìm gợi ý
        - rules_df (pd.DataFrame): Bảng luật kết hợp
        
        Trả về:
        - List[int]: Danh sách ID sản phẩm được gợi ý
        """
        recommendations = []
        
        for _, row in rules_df.iterrows():
            # antecedents và consequents là frozensets
            antecedents_set = row['antecedents']
            consequents_set = row['consequents']
            
            # Kiểm tra nếu product_id nằm trong antecedents
            # Chuyển antecedents_set thành list để kiểm tra
            if product_id in antecedents_set:
                # Thêm các sản phẩm trong consequents
                recommendations.extend(list(consequents_set))
        
        # Nếu không tìm thấy product_id trong antecedents, 
        # thử tìm trong consequents (sản phẩm được mua cùng)
        if not recommendations:
            for _, row in rules_df.iterrows():
                consequents_set = row['consequents']
                if product_id in consequents_set:
                    # Thêm các sản phẩm trong antecedents
                    recommendations.extend(list(row['antecedents']))
        
        return list(set(recommendations))  # Loại bỏ trùng lặp
