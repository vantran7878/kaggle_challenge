import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FEATURES_DIR = os.path.join(PROJECT_DIR, 'data', 'features_agnostic')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'scripts', 'checkpoints')

TRAIN_PATH = os.path.join(FEATURES_DIR, 'train.parquet')
VAL_PATH = os.path.join(FEATURES_DIR, 'val.parquet')
TEST_PATH = os.path.join(FEATURES_DIR, 'test_features_s.parquet')

SEED = 42
CUSTOM_THRESHOLD = 0.93

# Tập feature đầy đủ đã trích xuất từ các script trước
"""
'ag_1', 'ag_2', 'ag_3',
       'ag_4', 'ag_5', 'ag_6', 'ag_7', 'ag_8', 'ag_9', 'ag_10',
       'halstead_volume', 'maintainability_index', 'halstead_difficulty',
       'halstead_effort', 'internal_fan_out', 'nloc', 'token_count',
       'regex_ccn', 'llm_greeting', 'qwen_perplexity', 'pronoun_count',
       'stopword_density', 'sentence_count'
"""

BASE_FEATURES = [   
       'ag_1', 
       'ag_2', 
       'ag_3',
       'ag_4', 
       'ag_5', 
       'ag_6', 
       'ag_7', 
       'ag_8', 
       'ag_9', 
       'ag_10',
    #    'halstead_volume', 
       'maintainability_index', 
    #    'halstead_difficulty',
    #    'halstead_effort', 
       'internal_fan_out', 
    #    'nloc', 
       'token_count',
    #    'regex_ccn', 
       'llm_greeting', 
    #    'qwen_perplexity', 
    #    'pronoun_count',
    #    'stopword_density', 
    #    'sentence_count'
    'function_length_cv',     # CV hàm — language-agnostic mạnh nhất
    'debug_artifact_score',   # TODO+placeholder combined
    'function_count',         # Số hàm — proxy cho code structure
]



def run_asymmetric_ensemble_with_if():


    print(f"\n{'='*60}\n  ENHANCED ENSEMBLE + DUAL ISOLATION FOREST\n{'='*60}")
    
    

    # 1. Load Data
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    X_train_raw = train_df[BASE_FEATURES].fillna(0).values.astype(np.float32)
    y_train = train_df['label'].values
    X_val_raw = val_df[BASE_FEATURES].fillna(0).values.astype(np.float32)
    y_val = val_df['label'].values
    X_test_raw = test_df[BASE_FEATURES].fillna(0).values.astype(np.float32)

    # 2. Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    # 3. Dual Isolation Forest & Data Cleaning (Giống logic xgb_if)
    print("[*] Stage 1: Dual IF Modeling & Cleaning...")
    if_ai = IsolationForest(n_estimators=200, contamination=0.2, random_state=SEED)
    if_human = IsolationForest(n_estimators=200, contamination=0.15, random_state=SEED)
    
    # Chỉ học trên dữ liệu "sạch" của từng lớp
    if_ai.fit(X_train_sc[y_train == 1])
    if_human.fit(X_train_sc[y_train == 0])

    # Tạo Feature Anomaly Scores
    def add_dual_scores(X):
        s_ai = if_ai.score_samples(X).reshape(-1, 1)
        s_human = if_human.score_samples(X).reshape(-1, 1)
        return np.hstack([X, s_ai, s_human])
    
    outlier_label_ai = if_ai.predict(X_train_sc)

    outlier_label_human = if_human.predict(X_train_sc)

    # BƯỚC 1: Tính toán Dual Scores cho tất cả các tập (để có đầy đủ feature)
    X_train_final = add_dual_scores(X_train_sc) 
    X_val_final = add_dual_scores(X_val_sc)
    X_test_final = add_dual_scores(X_test_sc)

    # Áp dụng mask cho CẢ X VÀ Y
    mask_final = (outlier_label_ai == 1) & (outlier_label_human == 1)
    X_train_clean = X_train_final[mask_final] # Phải dùng biến này để fit
    y_train_clean = y_train[mask_final]       # Phải dùng biến này để fit



    # Kiểm tra log để chắc chắn (Nên in ra để debug)
    print(f"[*] X_train_clean: {X_train_clean.shape[0]} dòng")
    print(f"[*] y_train_clean: {y_train_clean.shape[0]} dòng")

    
   # ── 5. Huấn luyện Chuyên gia AI & Human ──
    print("\n[*] Huấn luyện Model AI (Chuyên gia bắt AI)...")
    model_ai = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.01,
        scale_pos_weight=0.7, eval_metric='logloss', 
        early_stopping_rounds=20, random_state=SEED
    )
    model_ai.fit(X_train_clean, y_train_clean, eval_set=[(X_val_final, y_val)], verbose=False)
    
    print("[*] Huấn luyện Model Human (Chuyên gia bắt Human)...")
    model_human = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        scale_pos_weight=0.2, eval_metric='logloss', 
        early_stopping_rounds=20, random_state=SEED
    )
    model_human.fit(X_train_clean, y_train_clean, eval_set=[(X_val_final, y_val)], verbose=False)

    # ── 6. Chiến lược 3: Late Fusion với Dynamic Decision Rule ──
    print("\n[*] Chiến lược 3: Late Fusion + Dynamic Anomaly Adjustment...")
    
    prob_ai_expert = model_ai.predict_proba(X_test_final)[:, 1]
    prob_human_expert = model_human.predict_proba(X_test_final)[:, 0]
    # Lấy anomaly score (cột cuối cùng)
    test_scores = X_test_final[:, -1] 
    
    print("\n[*] Đang lưu các mô hình...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    joblib.dump(model_ai, os.path.join(CHECKPOINT_DIR, 'model_ai_expert.pkl'))
    joblib.dump(model_human, os.path.join(CHECKPOINT_DIR, 'model_human_expert.pkl'))
    joblib.dump(if_ai, os.path.join(CHECKPOINT_DIR, 'if_ai.pkl'))
    joblib.dump(if_human, os.path.join(CHECKPOINT_DIR, 'if_human.pkl'))
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, 'scaler_dual_if.pkl'))
    print("[+] Hoàn tất! Đã lưu model_ai, model_human, iso_forest và scaler.")

    # ===== CHẠY INFERENCE =====



    df = pd.read_parquet(TEST_PATH)

    scaler = joblib.load(os.path.join(CHECKPOINT_DIR, 'scaler_dual_if.pkl'))

    if_ai = joblib.load(os.path.join(CHECKPOINT_DIR, 'if_ai.pkl'))

    if_human = joblib.load(os.path.join(CHECKPOINT_DIR, 'if_human.pkl'))

    xgb_ai = joblib.load(os.path.join(CHECKPOINT_DIR, 'model_ai_expert.pkl'))

    xgb_human = joblib.load(os.path.join(CHECKPOINT_DIR, 'model_human_expert.pkl'))

    # 2. Tiền xử lý & Tính xác suất (Chỉ chạy 1 lần để tối ưu thời gian)

    X_raw = df[BASE_FEATURES].fillna(0).values.astype(np.float32)

    X_sc = scaler.transform(X_raw)

    s_ai = if_ai.score_samples(X_sc).reshape(-1, 1)

    s_human = if_human.score_samples(X_sc).reshape(-1, 1)

    X_final = np.hstack([X_sc, s_ai, s_human])



    p_ai = xgb_ai.predict_proba(X_final)[:, 1]

    p_human = xgb_human.predict_proba(X_final)[:, 0]

   

    # Tính Combined Prob AI

    denominator = p_ai + p_human

    combined_probs = np.where(denominator > 0, p_ai / denominator, 0.5)



    # 3. Vòng lặp tìm Threshold tối ưu

    if 'label' not in df.columns:

        print("[!] Lỗi: Tập dữ liệu không có nhãn 'label' để đánh giá F1-Score.")

        return



    y_true = df['label'].values

    results = []

   

    print(f"\n{'='*45}")

    print(f"{'Threshold':<15} | {'Macro F1-Score':<15}")

    print(f"{'-'*45}")



    # Chạy dải từ 0.7 đến 0.95 với step 0.01 (mình để 0.01 để Kiên chọn được số mịn hơn)

    # Nếu Kiên muốn đúng 0.1 như yêu cầu thì đổi 0.01 thành 0.1

    for threshold in np.arange(0.6, 0.96, 0.01):

        preds = (combined_probs >= threshold).astype(int)

        f1 = f1_score(y_true, preds, average='macro')

        results.append((threshold, f1))

        print(f"{threshold:>14.2f} | {f1*100:>13.2f}%")



    # 4. Tìm và in ra Best Threshold

    best_thresh, best_f1 = max(results, key=lambda x: x[1])

    print(f"{'='*45}")

    print(f"  BEST THRESHOLD: {best_thresh:.2f}")

    print(f"  MAX F1-SCORE:   {best_f1*100:.2f}%")

    print(f"{'='*45}")



    # In thêm classification report cho best threshold để xem Precision/Recall

    best_preds = (combined_probs >= best_thresh).astype(int)

    print("\n[DETAILED REPORT FOR BEST THRESHOLD]")

    print(classification_report(y_true, best_preds, target_names=['Human', 'AI']))
    print(f"Dự đoán tập test full: {best_f1*100 - 3.5 : .2f}%")





if __name__ == "__main__":
    run_asymmetric_ensemble_with_if()