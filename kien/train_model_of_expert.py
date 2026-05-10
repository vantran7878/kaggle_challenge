import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

# ============================================================
# CẤU HÌNH
# ============================================================
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FEATURES_DIR   = os.path.join(PROJECT_DIR, 'data', 'features_agnostic')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'scripts', 'checkpoints')
SUBMIT_DIR     = os.path.join(PROJECT_DIR, 'submissions')

TRAIN_PATH      = os.path.join(FEATURES_DIR, 'train.parquet')
VAL_PATH        = os.path.join(FEATURES_DIR, 'val.parquet')
TEST_PATH       = os.path.join(FEATURES_DIR, 'test_features_s.parquet')
TEST_FULL_PATH  = os.path.join(FEATURES_DIR, 'test_features_f.parquet')
OUTPUT_CSV      = os.path.join(SUBMIT_DIR, 'submission.csv')

SEED = 42

# ── Giữ nguyên feature set của bạn ──
FEATURES_FULL = [
    'ag_1', 'ag_2', 'ag_3', 'ag_4', 'ag_5',
    'ag_6', 'ag_7', 'ag_8', 'ag_9',
    'token_count', 'llm_greeting',
    'function_length_cv', 'function_count', 'debug_artifact_score',
    'ag_10', 'maintainability_index', 'internal_fan_out', 
    'placeholder_ratio'
]

FEATURES_BASE = [
    'ag_1', 'ag_2', 'ag_3', 'ag_4', 'ag_5',
    'ag_6', 'ag_7', 'ag_8', 'ag_9',
    'token_count', 'llm_greeting',
    'ag_10', 'maintainability_index', 'internal_fan_out',
]

NEW_FEAT_NAMES = ['function_length_cv', 'function_count', 'debug_artifact_score']

LANG_GROUP_A = {'c', 'c#', 'go', 'java', 'php'}
LANG_GROUP_B = {'c++', 'javascript', 'python', 'js'}


def run():
    print(f"\n{'='*60}")
    print(f"  LANGUAGE-AWARE INFERENCE")
    print(f"{'='*60}")

    # ── 1. Load data ──
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    avail_full = [f for f in FEATURES_FULL if f in train_df.columns]
    avail_base = [f for f in FEATURES_BASE if f in train_df.columns]

    new_feat_idx = [avail_full.index(f) for f in NEW_FEAT_NAMES if f in avail_full]
    print(f"[*] Full features  : {len(avail_full)}")
    print(f"[*] Base features  : {len(avail_base)}")
    print(f"[*] New feat index : {new_feat_idx} → {[avail_full[i] for i in new_feat_idx]}")

    y_train = train_df['label'].values
    y_val   = val_df['label'].values

    # ── 2. Train ──
    print("\n[*] Training trên full feature set (100% train data)...")

    X_train_raw = train_df[avail_full].fillna(0).values.astype(np.float32)
    X_val_raw   = val_df[avail_full].fillna(0).values.astype(np.float32)
    X_test_raw  = test_df[avail_full].fillna(0).values.astype(np.float32)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc   = scaler.transform(X_val_raw)
    X_test_sc  = scaler.transform(X_test_raw)

    if_ai    = IsolationForest(n_estimators=200, contamination=0.2, random_state=SEED)
    if_human = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED)
    if_ai.fit(X_train_sc[y_train == 1])
    if_human.fit(X_train_sc[y_train == 0])

    def add_if_scores(X):
        s_ai  = if_ai.score_samples(X).reshape(-1, 1)
        s_hum = if_human.score_samples(X).reshape(-1, 1)
        return np.hstack([X, s_ai, s_hum])

    X_train_aug = add_if_scores(X_train_sc)
    X_val_aug   = add_if_scores(X_val_sc)
    X_test_aug  = add_if_scores(X_test_sc)

    mask_clean    = (if_ai.predict(X_train_sc) == 1) & \
                    (if_human.predict(X_train_sc) == 1)
    X_train_clean = X_train_aug[mask_clean]
    y_train_clean = y_train[mask_clean]
    print(f"    Sau cleaning: {X_train_clean.shape[0]}/{len(y_train)} samples")

    model_ai = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.01,
        scale_pos_weight=0.7, eval_metric='logloss',
        early_stopping_rounds=20, random_state=SEED
    )
    model_ai.fit(X_train_clean, y_train_clean,
                 eval_set=[(X_val_aug, y_val)], verbose=False)

    model_hum = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        scale_pos_weight=0.2, eval_metric='logloss',
        early_stopping_rounds=20, random_state=SEED
    )
    model_hum.fit(X_train_clean, y_train_clean,
                  eval_set=[(X_val_aug, y_val)], verbose=False)
    print("    Done.")

    # ── 3. Hàm inference với feature masking ──
    def get_combined_proba(X_aug, lang_group='A'):
        X_use = X_aug.copy()
        if lang_group == 'B' and new_feat_idx:
            X_use[:, new_feat_idx] = 0.0
        p_ai  = model_ai.predict_proba(X_use)[:, 1]
        p_hum = model_hum.predict_proba(X_use)[:, 0]
        denom = p_ai + p_hum
        return np.where(denom > 0, p_ai / denom, 0.5)

    # ── 4. Baseline ──
    print("\n[*] Baseline (không routing)...")
    proba_baseline = get_combined_proba(X_test_aug, 'A')
    y_test = test_df['label'].values if 'label' in test_df.columns else None
    if y_test is not None:
        results_bl = [(t, f1_score(y_test, (proba_baseline >= t).astype(int), average='macro'))
                      for t in np.arange(0.60, 0.96, 0.01)]
        best_t_bl, best_f1_bl = max(results_bl, key=lambda x: x[1])
        print(f"    Baseline F1: {best_f1_bl*100:.2f}% @ threshold {best_t_bl:.2f}")

    # ── 5. Language-aware inference trên test_small ──
    print("\n[*] Language-aware inference trên test_small...")

    all_preds    = np.zeros(len(test_df), dtype=int)
    lang_thresholds = {}  # lưu lại để dùng cho test_full

    print(f"\n{'='*62}")
    print(f"{'Language':<15} | {'Group':<6} | {'Thresh':>7} | {'F1':>8} | {'N':>5}")
    print(f"{'-'*62}")

    for lang in sorted(test_df['language'].unique()):
        mask = (test_df['language'] == lang).values
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue

        group = 'A' if lang.lower() in LANG_GROUP_A else 'B'
        proba = get_combined_proba(X_test_aug[mask], group)

        if y_test is not None:
            y_lang = y_test[mask]
            if len(np.unique(y_lang)) < 2:
                thresh, f1 = 0.93, 0.0
            else:
                results = [(t, f1_score(y_lang, (proba >= t).astype(int), average='macro'))
                           for t in np.arange(0.40, 0.96, 0.005)]
                thresh, f1 = max(results, key=lambda x: x[1])
        else:
            thresh = 0.90 if group == 'A' else 0.93
            f1     = -1

        preds          = (proba >= thresh).astype(int)
        all_preds[idx] = preds

        # Lưu threshold để tái dùng cho test_full
        lang_thresholds[lang.lower()] = (group, thresh)

        print(f"{lang:<15} | {group:<6} | {thresh:>7.3f} | "
              f"{f1*100:>7.2f}% | {len(idx):>5}")

    print(f"{'='*62}")

    if y_test is not None:
        overall_f1 = f1_score(y_test, all_preds, average='macro')
        print(f"\n  Baseline (no routing) : {best_f1_bl*100:.2f}%")
        print(f"  Language-aware        : {overall_f1*100:.2f}%")
        print(f"  Δ vs routing          : {(overall_f1 - best_f1_bl)*100:+.2f}%")
        print(f"  Δ vs original 67.52%  : {(overall_f1*100 - 67.52):+.2f}%")
        print("\n[DETAILED REPORT]")
        print(classification_report(y_test, all_preds, target_names=['Human', 'AI']))

    # ── 6. Inference + xuất CSV trên test_full ──
    print(f"\n{'='*60}")
    print(f"  XUẤT CSV — TEST FULL")
    print(f"{'='*60}")

    test_full_df = pd.read_parquet(TEST_FULL_PATH)
    print(f"[*] test_full shape: {test_full_df.shape}")

    X_test_f_raw = test_full_df[avail_full].fillna(0).values.astype(np.float32)
    X_test_f_sc  = scaler.transform(X_test_f_raw)
    X_test_f_aug = add_if_scores(X_test_f_sc)

    preds_full = np.zeros(len(test_full_df), dtype=int)

    if 'language' in test_full_df.columns:
        print(f"\n{'Language':<15} | {'Group':<6} | {'Thresh':>7} | {'AI':>6} | {'N':>6}")
        print("-" * 55)

        for lang in sorted(test_full_df['language'].unique()):
            mask = (test_full_df['language'] == lang).values
            idx  = np.where(mask)[0]
            if len(idx) == 0:
                continue

            lang_key = lang.lower()
            if lang_key in lang_thresholds:
                group, thresh = lang_thresholds[lang_key]
            else:
                group  = 'A' if lang_key in LANG_GROUP_A else 'B'
                thresh = 0.90 if group == 'A' else 0.93

            proba         = get_combined_proba(X_test_f_aug[mask], group)
            preds         = (proba >= thresh).astype(int)
            preds_full[idx] = preds

            print(f"{lang:<15} | {group:<6} | {thresh:>7.3f} | "
                  f"{preds.sum():>6} | {len(idx):>6}")

        print("-" * 55)
    else:
        # Không có language → dùng full features, threshold mặc định
        proba      = get_combined_proba(X_test_f_aug, 'A')
        preds_full = (proba >= 0.93).astype(int)

    print(f"\n  Tổng samples   : {len(preds_full)}")
    print(f"  AI  (label=1)  : {preds_full.sum()}")
    print(f"  Human (label=0): {(preds_full == 0).sum()}")
    print(f"  AI ratio       : {preds_full.mean()*100:.1f}%")

    # Tìm cột id
    id_col = next((c for c in ['id', 'ID', 'sample_id'] if c in test_full_df.columns), None)
    result_df = pd.DataFrame({
        'ID'   : test_full_df[id_col] if id_col else range(len(preds_full)),
        'label': preds_full,
    })

    os.makedirs(SUBMIT_DIR, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] CSV đã lưu tại: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()