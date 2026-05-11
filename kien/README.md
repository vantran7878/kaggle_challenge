# Báo Cáo Phương Pháp: Phát Hiện Code AI (AI vs Human)

Tài liệu này tổng hợp lại toàn bộ phương pháp tiếp cận, danh sách các đặc trưng (features) và các siêu tham số (hyperparameters) đã được sử dụng trong bài toán phân loại mã nguồn được tạo bởi AI và Humans. Tài liệu này bao gồm mô tả cho cả hai script: `train_dual_if_asymmetric.py` (bản gốc cốt lõi) và `train_model_of_expert.py` (phiên bản mở rộng có phân loại theo ngôn ngữ).

---

## 1. Phương Pháp Tiếp Cận (Methodology)

Bài toán sử dụng một pipeline kết hợp giữa **Làm sạch dữ liệu ngoại lai (Anomaly Detection)** và **Học máy kết hợp bất đối xứng (Asymmetric Ensemble / Mixture of Experts)**. Trong phiên bản nâng cao, hệ thống còn kết hợp thêm phân loại nhận biết ngôn ngữ (**Language-aware Inference**). Các bước cụ thể:

### 1.1 Khử nhiễu và Sinh đặc trưng phụ với Dual Isolation Forest
- Huấn luyện hai mô hình **Isolation Forest** chia theo nhãn:
  - Một mô hình chỉ học trên dữ liệu code AI (`y=1`).
  - Một mô hình chỉ học trên dữ liệu code của con người (`y=0`).
- **Mục đích 1 (Data Cleaning):** Lọc bỏ các mẫu "outlier" (nhiễu, khó phân loại) khỏi tập train (chỉ giữ lại các sample được cả 2 mô hình coi là inlier) giúp tập huấn luyện sạch hơn.
- **Mục đích 2 (Feature Engineering):** Trích xuất `score_samples` từ 2 mô hình này (`s_ai`, `s_human`) ráp ngược lại làm đặc trưng đầu vào (augmented features) cho model phân loại chính.

### 1.2 "Chuyên gia" Bất Đối Xứng (Asymmetric XGBoost Experts)
Thay vì dùng 1 mô hình XGBoost chung, phương pháp này huấn luyện 2 mô hình (Experts) chuyên biệt:
- **Model AI Expert:** Tập trung bắt các pattern của AI (Sử dụng `scale_pos_weight` cao hơn để nhạy cảm với class AI).
- **Model Human Expert:** Tập trung bắt các pattern của người (Sử dụng `scale_pos_weight` thấp hơn).

### 1.3 Kế hợp trễ (Late Fusion)
- **Late Fusion:** Cả hai file đều sử dụng chiến lược dự đoán dựa trên tổng hợp xác suất, trong đó xác suất từ 2 models được chuẩn hóa thông qua công thức tỉ lệ: $\frac{P_{ai}}{P_{ai} + P_{human}}$
  - Trong `train_dual_if_asymmetric.py`: Khi đưa ra kết quả, mô hình sử dụng một ngưỡng (Threshold) cố định tĩnh (ví dụ: `0.93`) hoặc một ngưỡng dò tìm chung cho toàn bộ tập dữ liệu.
  
### 1.4 Nhận diện theo ngôn ngữ (Chỉ có trong `train_model_of_expert.py`)
- **Masking theo nhóm ngôn ngữ:** Dữ liệu chia làm 2 nhóm (Group A: C, C#, Go, Java, PHP và Group B: C++, JS, Python). Đối với một số ngôn ngữ (Group B), các đặc trưng mới được đưa về `0.0` (masking) để tránh overfitting hay lệch phân phối (distribution shift).
- **Dynamic Thresholding:** Tìm giá trị Threshold tối ưu (F1-score cao nhất) riêng biệt cho từng ngôn ngữ và nhóm ngôn ngữ (thay vì dùng một Global Threshold như bản `train_dual_if_asymmetric`).

---

## 2. Các Đặc Trưng Được Sử Dụng (Features)

Tập đặc trưng có sự thay đổi nhẹ giữa 2 phiên bản:

- **Nhóm Meta/Statistical Features (ag_1 đến ag_10):** Chứa các thông tin ẩn/đặc trưng meta từ Data - từ repo của Giovanni.
- **Nhóm Mã nguồn & Cấu trúc (Code Metrics):**
  - `token_count`: Số lượng token.
  - `function_length_cv`: Hệ số biến thiên chiều dài các hàm.
  - `function_count`: Tổng số hàm.
  - `internal_fan_out`: Độ phức tạp gọi hàm (tương tác giữa các modules).
  - `maintainability_index`: Chỉ số dễ bảo trì của code.
  - `placeholder_ratio`: Tỉ lệ sử dụng placeholder text trong code **(Chỉ có ở `train_model_of_expert.py`)**.
- **Nhóm Dấu hiệu AI (AI-specific heuristics):**
  - `llm_greeting`: Dấu hiệu có lời chào đặc trưng của LLM (VD: "Here is the code...").
  - `debug_artifact_score`: Điểm đánh giá các dư lượng text debug.
- **Nhóm Augmented (Từ Isolation Forest):**
  - `s_ai` & `s_human`: Anomaly scores ráp vào cuối quy trình.

---

## 3. Cấu Hình Tham Số (Hyperparameters)

Các mô hình được config bằng các Hyperparameter cụ thể nhằm phù hợp với kỹ thuật "chuyên gia":

### 3.1 Dual Isolation Forest
- **IF (chuyên AI):** Cả hai file đều sử dụng chung thông số này.
  - `n_estimators`: 200
  - `contamination`: 0.20
  - `random_state`: 42
- **IF (chuyên Human):** 
  - `n_estimators`: 200
  - `random_state`: 42
  - `contamination`: **0.15** (trong `train_dual_if_asymmetric.py`) hoặc **0.10** (trong `train_model_of_expert.py`).

### 3.2 Dual XGBoost Experts
| Tham số | Model AI Expert | Model Human Expert |
| :--- | :--- | :--- |
| **Mục đích** | Bắt code AI | Bắt code người |
| `n_estimators` | 300 | 400 |
| `max_depth` | 5 | 7 |
| `learning_rate` | 0.01 | 0.05 |
| `scale_pos_weight` | 0.7 | 0.2 |
| `eval_metric` | logloss | logloss |
| `early_stopping_rounds`| 20 | 20 |

### 3.3 Ngưỡng quyết định (Thresholds)
Ngưỡng tối ưu để phân loại được tinh chỉnh động, nhưng giá trị baseline fallback (khi không optimize được bằng nhãn) cho các nhóm:
- **Group A** (C, C#, Go, Java, PHP): `Threshold = 0.90`
- **Group B** (C++, JS, Python): `Threshold = 0.93`
- _(Ngưỡng có thể được dò cụ thể trong dải `[0.40 : 0.96]` đối với tập validation có nhãn)_
