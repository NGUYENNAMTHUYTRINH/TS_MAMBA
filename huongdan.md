# Huong dan cai dat (WSL + Mamba)

## 1) Cai WSL va dang nhap

- Chay lenh:

```bash
wsl --install -d Ubuntu
```

- Tao user va dang nhap theo huong dan tren man hinh.

## 2) Tao moi truong ao Python 3.10

```bash
# Ve thu muc Home
cd ~

# Xoa moi truong cu neu bi loi
rm -rf mamba_env

# Tao moi truong ao (ep Python 3.10)
python3.10 -m venv mamba_env

# Kich hoat moi truong ao
source mamba_env/bin/activate
```

## 3) Chuyen den thu muc du an

```bash
cd /mnt/d/KLTN/TS_MAMBA
```

## 4) Cai PyTorch

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://pytorch.org
```

## 5) Cai CUDA Toolkit tren WSL

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

## 6) Cai cac goi can thiet

```bash
# Cai causal-conv1d
pip install causal-conv1d>=1.4.0 --no-build-isolation

# Cai mamba-ssm goc tu tac gia
pip install mamba-ssm --no-build-isolation

# Nang cap torchvision va torchaudio
pip install --upgrade torchvision torchaudio
```

## 7) Mo project trong VS Code (WSL)

- Cai extension WSL cua Microsoft (extension dau tien, ~39M luot tai).
- Dat Ubuntu lam WSL mac dinh de tranh xung dot voi WSL cua Docker:

```bash
wsl --set-default Ubuntu
```

- Trong VS Code:
  - Bam vao bieu tuong `><` o goc duoi ben trai.
  - Chon "Connect to WSL".
  - Chon "Open Folder" va dan duong dan:

```
/mnt/d/KLTN/TS_MAMBA/
```


source mamba_env/bin/activate
cd /mnt/d/KLTN/TS_MAMBA
./lnx_venv/bin/python -m pip install -r requirements.txt

NẾu lỗi thì gỡ torch cũ

./lnx_venv/bin/python -m pip show torch #( xem đường dẫn)

#nhớ xóa đúng đường dẫn
rm -rf /mnt/d/KLTN/TS_MAMBA/lnx_venv/lib/python3.14/site-packages/torch*
rm -rf /mnt/d/KLTN/TS_MAMBA/lnx_venv/lib/python3.14/site-packages/*torch*

cài lại
./lnx_venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

chạy 
./lnx_venv/bin/python mamba/train_mamba_aqi.py \
  --data-path dataset/air_quality.csv \
  --epochs 10 \
  --window-size 72 \
  --horizon 12 \
  --batch-size 128 \
  --device cuda \
  --amp