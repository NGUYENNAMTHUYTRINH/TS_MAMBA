# iTransformer — Air Quality Forecasting

> **Dự báo chất lượng không khí đa biến** sử dụng kiến trúc **iTransformer** (Inverted Transformer), được tùy biến từ paper gốc [iTransformer (arXiv 2310.06625)](https://arxiv.org/abs/2310.06625) cho bài toán chuỗi thời gian chất lượng không khí đa trạm quan trắc.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Dữ liệu](#-dữ-liệu)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Tham số cấu hình](#-tham-số-cấu-hình)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [License](#-license)

---

## 🌐 Tổng quan

Dự án này áp dụng kiến trúc **iTransformer** để dự báo chất lượng không khí tại nhiều trạm quan trắc đồng thời. Điểm đặc trưng của iTransformer là **đảo ngược chiều attention**: thay vì áp dụng self-attention theo chiều thời gian (time-steps), mô hình nhúng toàn bộ chuỗi thời gian của từng biến thành một token duy nhất và áp dụng attention theo chiều biến (**variate-wise attention**). Điều này giúp mô hình nắm bắt tốt hơn mối tương quan giữa các biến trong dự báo đa biến.

### Đặc điểm nổi bật

| Tính năng | Mô tả |
|-----------|-------|
| **Kiến trúc** | iTransformer — Encoder-only với inverted embedding |
| **Dữ liệu** | Chất lượng không khí đa trạm, đa chỉ số (`pm25`, `pm10`, `no2`, `o3`, `so2`, `co`, `aod`, `dust`, `uv_index`, `co2`, `aqi`) |
| **Chế độ dự báo** | M (multivariate → multivariate), S (univariate), MS (multivariate → univariate) |
| **Chuẩn hóa** | Instance Normalization (Non-stationary Transformer style) |
| **Embedding thời gian** | TimeFeature (`timeF`), Fixed, hoặc Learned |
| **Baseline so sánh** | Vanilla Transformer (Encoder-Decoder) |

---

## 🏗 Kiến trúc mô hình

```
Input x_enc  [B, L, N]
      │
      ▼
Instance Normalization  (tùy chọn, use_norm=1)
      │
      ▼
DataEmbedding_inverted  (B, L, N) ──permute──► (B, N, L) ──Linear──► (B, N, d_model)
  ┌─ Nhúng giá trị chuỗi (Linear: seq_len → d_model)
  └─ Có thể kết hợp time covariates làm token bổ sung
      │
      ▼
Encoder  (e_layers lớp)
  ├─ Multi-Head Full Attention  (attention theo chiều variate)
  ├─ Add & Norm (LayerNorm)
  └─ Feed-Forward (d_model → d_ff → d_model) + GELU + Dropout
      │
      ▼
Projector  (Linear: d_model → pred_len)
  (B, N, d_model) ──► (B, N, pred_len) ──permute──► (B, pred_len, N)
      │
      ▼
De-normalization  (phục hồi mean & std từ x_enc)
      │
      ▼
Output  [B, pred_len, N]
```

**iTransformer vs Transformer truyền thống:**

| Thành phần | Transformer | iTransformer |
|---|---|---|
| Token | Mỗi time-step là 1 token | Mỗi biến (variate) là 1 token |
| Embedding | `(B, L, N)` → `(B, L, d_model)` | `(B, L, N)` → `(B, N, d_model)` |
| Attention | Temporal (L × L) | Variate-wise (N × N) |
| Decoder | Có | Không (encoder-only) |

---

## 📁 Cấu trúc thư mục

```
iTransformer/
├── run.py                          # Entry point chính — định nghĩa toàn bộ tham số CLI
│
├── scripts/
│   └── air_quality_forecast.sh    # Script bash chạy thí nghiệm M và S
│
├── experiments/
│   ├── exp_basic.py               # Base class: khởi tạo device, model dict
│   └── exp_long_term_forecasting.py  # Train / Validate / Test / Predict pipeline
│
├── model/
│   ├── iTransformer.py            # Kiến trúc iTransformer (encoder-only, inverted)
│   └── Transformer.py             # Baseline Transformer (encoder-decoder)
│
├── layers/
│   ├── Embed.py                   # DataEmbedding, DataEmbedding_inverted, TimeFeatureEmbedding
│   ├── SelfAttention_Family.py    # FullAttention, AttentionLayer
│   └── Transformer_EncDec.py     # Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer
│
├── data_provider/
│   ├── data_factory.py            # Registry dataset + DataLoader factory
│   └── data_loader.py             # Dataset_AirQuality — pivot, split, scale, timestamp
│
├── utils/
│   ├── metrics.py                 # MAE, MSE, RMSE, MAPE, MSPE
│   ├── tools.py                   # EarlyStopping, adjust_learning_rate, visual
│   ├── timefeatures.py            # Trích xuất time features (hour, weekday, ...)
│   └── masking.py                 # Triangular causal mask
│
├── dataset/
│   └── air_quality/
│       └── air_quality.csv        # Dữ liệu đầu vào (không được commit vào git)
│
├── checkpoints/                   # Model checkpoint được lưu tại đây
├── results/                       # .npy kết quả dự báo
├── test_results/                  # Biểu đồ PDF từng batch kiểm tra
├── result_long_term_forecast.txt  # Log MSE/MAE của các thí nghiệm
│
└── requirements.txt               # Python dependencies
```

---

## 📊 Dữ liệu

### Định dạng file `air_quality.csv`

File CSV cần có cấu trúc **long-format** (mỗi dòng = 1 trạm × 1 thời điểm):

```
ts_utc,pm25,pm10,no2,o3,so2,co,aod,dust,uv_index,co2,aqi,location_key
2024-01-01 00:00:00+00:00,15.2,28.1,12.5,...,45,VN-HAN-001
2024-01-01 00:00:00+00:00,22.7,41.3,18.9,...,67,VN-HAN-002
2024-01-01 01:00:00+00:00,16.1,29.4,13.1,...,48,VN-HAN-001
...
```

| Cột | Mô tả |
|-----|-------|
| `ts_utc` | Timestamp UTC (ISO 8601) |
| `pm25`, `pm10`, `no2`, `o3`, `so2`, `co` | Chỉ số ô nhiễm không khí |
| `aod`, `dust`, `uv_index`, `co2` | Các chỉ số môi trường bổ sung |
| `aqi` | Air Quality Index (chỉ số tổng hợp) |
| `location_key` | Mã định danh trạm quan trắc |

### Cấu hình mặc định (23 trạm × 11 chỉ số)

- **Số biến (enc_in):** `23 locations × 17 features = 391` (chế độ M)  
  hoặc `23` (chế độ S, chỉ dự báo AQI)
- **Split:** 70% train / 10% val / 20% test (theo thứ tự thời gian)
- **Minimum time-steps:** ≥ `seq_len + pred_len` (mặc định ≥ 192)

---

## 🛠 Cài đặt

### Yêu cầu hệ thống

- Python ≥ 3.8
- CUDA ≥ 11.7 (khuyến nghị, có thể chạy CPU)

### Cài đặt dependencies

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Cài đặt packages
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas==1.5.3
scikit-learn==1.2.2
numpy==1.23.5
matplotlib==3.7.0
torch==2.0.0
reformer-pytorch==1.4.4
```

---

## 🚀 Sử dụng

### 1. Chuẩn bị dữ liệu

Đặt file `air_quality.csv` vào thư mục:
```
dataset/air_quality/air_quality.csv
```

### 2. Training

#### Dùng script bash (khuyến nghị)

```bash
bash ./scripts/air_quality_forecast.sh
```

Script sẽ chạy **2 thí nghiệm** liên tiếp:
- **M (Multivariate):** Dự báo tất cả 391 biến (23 trạm × 17 chỉ số)
- **S (Univariate per location):** Chỉ dự báo AQI tại 23 trạm

#### Dùng `run.py` trực tiếp

**Chế độ M (multivariate → multivariate):**
```bash
python run.py \
  --is_training 1 \
  --model_id air_quality_M \
  --model iTransformer \
  --data air_quality \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 391 --dec_in 391 --c_out 391 \
  --e_layers 3 \
  --d_model 512 --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --patience 3 \
  --use_norm 1
```

**Chế độ S (chỉ AQI):**
```bash
python run.py \
  --is_training 1 \
  --model_id air_quality_AQI \
  --model iTransformer \
  --data air_quality \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --features S \
  --target aqi \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 23 --dec_in 23 --c_out 23 \
  --e_layers 3 \
  --d_model 512 --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 10
```

**Tìm cấu hình tối ưu (ví dụ seq_len=168, pred_len=24):**
```bash
python run.py \
  --is_training 1 \
  --model_id aqi_gpu_best \
  --model iTransformer \
  --data air_quality \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --features S --target aqi \
  --seq_len 168 --label_len 72 --pred_len 24 \
  --enc_in 23 --dec_in 23 --c_out 23 \
  --e_layers 4 \
  --d_model 512 --d_ff 1024 \
  --n_heads 8 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 10
```

### 3. Testing (chỉ inference)

```bash
python run.py \
  --is_training 0 \
  --model_id air_quality_AQI \
  --model iTransformer \
  --data air_quality \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --features S --target aqi \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 23 --dec_in 23 --c_out 23 \
  --e_layers 3 --d_model 512 --d_ff 512
```

### 4. So sánh với Baseline Transformer

```bash
python run.py \
  --is_training 1 \
  --model_id air_quality_AQI \
  --model Transformer \
  --data air_quality \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --features S --target aqi \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 23 --dec_in 23 --c_out 23 \
  --e_layers 3 --d_model 512 --d_ff 512
```

---

## ⚙️ Tham số cấu hình

### Tham số cơ bản

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--is_training` | `1` | `1` = train+test, `0` = test only |
| `--model_id` | `air_quality` | Tên định danh thí nghiệm |
| `--model` | `iTransformer` | Kiến trúc: `iTransformer` hoặc `Transformer` |

### Tham số dữ liệu

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data` | `air_quality` | Loại dataset |
| `--root_path` | `./dataset/air_quality/` | Thư mục chứa data |
| `--data_path` | `air_quality.csv` | Tên file CSV |
| `--features` | `M` | `M`: multi→multi, `S`: uni, `MS`: multi→uni |
| `--target` | `aqi` | Biến mục tiêu (cho chế độ S/MS) |
| `--freq` | `h` | Tần suất thời gian: `s/t/h/d/b/w/m` |

### Tham số dự báo

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--seq_len` | `96` | Độ dài chuỗi đầu vào (look-back window) |
| `--label_len` | `48` | Token bắt đầu cho decoder |
| `--pred_len` | `96` | Độ dài dự báo |

### Tham số kiến trúc

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--enc_in` | `391` | Số biến đầu vào encoder |
| `--dec_in` | `391` | Số biến đầu vào decoder |
| `--c_out` | `391` | Số biến đầu ra |
| `--d_model` | `512` | Chiều embedding |
| `--n_heads` | `8` | Số attention head |
| `--e_layers` | `3` | Số lớp encoder |
| `--d_layers` | `1` | Số lớp decoder (chỉ cho Transformer) |
| `--d_ff` | `512` | Chiều Feed-Forward |
| `--dropout` | `0.1` | Tỉ lệ dropout |
| `--embed` | `timeF` | Kiểu time embedding: `timeF/fixed/learned` |
| `--use_norm` | `1` | Bật instance normalization |

### Tham số tối ưu

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--train_epochs` | `10` | Số epoch |
| `--batch_size` | `16` | Batch size |
| `--patience` | `3` | Patience early stopping |
| `--learning_rate` | `0.0005` | Learning rate ban đầu |
| `--lradj` | `type1` | Chiến lược điều chỉnh LR |
| `--use_amp` | `False` | Dùng mixed precision (AMP) |

### Tham số GPU

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--use_gpu` | `True` | Sử dụng GPU nếu có |
| `--gpu` | `0` | GPU device ID |
| `--use_multi_gpu` | `False` | DataParallel trên nhiều GPU |
| `--devices` | `0,1,2,3` | Danh sách GPU IDs |

---

## 📈 Kết quả thực nghiệm

Kết quả được ghi tự động vào `result_long_term_forecast.txt`. Dưới đây là tổng hợp các thí nghiệm đã chạy:

### Dự báo AQI (features=S, seq_len=96, pred_len=96)

| Model | MSE ↓ | MAE ↓ |
|-------|--------|--------|
| **iTransformer** | **0.5658** | **0.5508** |
| Transformer (baseline) | 0.7753 | 0.6691 |

### Dự báo AQI (features=S, seq_len=168, pred_len=24) — Best config

| Model | MSE ↓ | MAE ↓ |
|-------|--------|--------|
| **iTransformer** | **0.2614** | **0.3344** |

### Dự báo Multivariate (features=MS, seq_len=96, pred_len=96)

| Model | MSE ↓ | MAE ↓ |
|-------|--------|--------|
| iTransformer | 0.9255 | 0.7252 |

> **Nhận xét:** iTransformer vượt trội đáng kể so với Transformer truyền thống ở chế độ S. Cấu hình tối ưu `seq_len=168, pred_len=24` cho MSE thấp hơn ~2.3× so với cấu hình mặc định `96/96`.

### Outputs

Sau mỗi lần chạy, kết quả được lưu tại:

```
checkpoints/<setting>/checkpoint.pth     # Model weights tốt nhất
results/<setting>/metrics.npy           # [mae, mse, rmse, mape, mspe]
results/<setting>/pred.npy              # Dự báo
results/<setting>/true.npy              # Ground truth
test_results/<setting>/<i>.pdf          # Biểu đồ so sánh từng batch
result_long_term_forecast.txt           # Log tổng hợp MSE/MAE
```

---

## 🔧 Gợi ý tùy chỉnh

### Thêm dataset mới

1. Tạo class Dataset mới trong `data_provider/data_loader.py`
2. Đăng ký vào `data_dict` trong `data_provider/data_factory.py`
3. Thêm tham số `--data <tên_mới>` khi chạy

### Điều chỉnh `enc_in` theo dữ liệu thực tế

```
# features=M:
enc_in = số_trạm × số_chỉ_số_feature

# features=S:
enc_in = số_trạm
```

### Tăng hiệu suất

- Tăng `seq_len` (ví dụ 168 hoặc 336) để mô hình có nhiều context hơn
- Giảm `pred_len` (24 thay vì 96) để bài toán dễ hơn
- Tăng `e_layers` (3→4) và `d_ff` (512→1024) nếu có GPU mạnh
- Bật `--use_amp` để tăng tốc training với mixed precision

---

## 📄 License

Dự án này được phát hành dưới giấy phép [MIT License](LICENSE).

---

## 📚 Tham khảo

- **iTransformer paper:** [Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625) — Liu et al., ICLR 2024
- **Transformer paper:** [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Vaswani et al., NeurIPS 2017
- **Non-stationary Transformer:** Instance Normalization technique được sử dụng trong `use_norm`
