# KNN với bỏ phiếu có trọng số theo cosine (AG News)

## Tóm tắt

Dự án này trình bày một biến thể đơn giản của KNN cho bài toán phân loại văn bản: thay vì bỏ phiếu “đếm phiếu” (majority vote) trong $k$ láng giềng gần nhất, ta dùng **bỏ phiếu có trọng số theo độ tương đồng cosine**. Đặc trưng văn bản được mã hoá bằng TF‑IDF, và độ gần được tính bằng cosine similarity.

Mục tiêu chính: cung cấp một baseline KNN rõ ràng + một cải tiến nhẹ (weighted voting) để so sánh thực nghiệm trên AG News.

## Tài nguyên

- Notebook Colab (tham khảo/nguồn ý tưởng): https://colab.research.google.com/drive/1mNqr5vSHXIeetL-HozczCnukJTvro27M?usp=sharing
- Notebook trong repo: [KNN_Impove.ipynb](KNN_Impove.ipynb)

## Dataset

Sử dụng **AG News** (4 lớp): World, Sports, Business, Sci/Tech. Code tải qua Hugging Face Datasets và lấy một subset nhỏ để demo chạy nhanh.

## Phương pháp

### Biểu diễn văn bản

Văn bản được vector hoá bằng TF‑IDF với:

- `max_features = 5000`
- `ngram_range = (1, 2)`
- `stop_words = 'english'`

### Cosine similarity

Với 2 vector $u, v$:

$$
\mathrm{cos}(u, v) = \frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}
$$

Trong code, các vector được chuẩn hoá $\ell_2$ để việc tính cosine trở thành tích vô hướng.

### Baseline: KNN majority vote

Chọn tập $N_k(x)$ gồm $k$ mẫu train có cosine similarity lớn nhất với mẫu cần dự đoán $x$, sau đó dự đoán nhãn theo đa số:

$$
\hat{y} = \arg\max_{c} \sum_{i\in N_k(x)} \mathbf{1}(y_i=c)
$$

Triển khai tại module `KNNOptimized`.

### Đề xuất: KNN weighted vote theo cosine

Thay vì đếm phiếu, mỗi láng giềng đóng góp một trọng số bằng độ tương đồng cosine (clipping về $\ge 0$ để tránh điểm âm):

$$
\mathrm{score}(c) = \sum_{i\in N_k(x)} \max(\mathrm{cos}(x, x_i), 0)\,\mathbf{1}(y_i=c)
$$

$$
\hat{y} = \arg\max_{c}\ \mathrm{score}(c)
$$

Nếu cần xác suất, chuẩn hoá các score theo tổng score của mọi lớp.

Triển khai tại module `KNNWeightedCosine`.

## Tái lập thực nghiệm

### Cài đặt

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Chạy

```bash
python -m knn_weight.experiment
```

Tuỳ chọn:

```bash
python -m knn_weight.experiment --k 5 --train-size 8000 --test-size 600 --max-features 5000
```

Lưu ý: Implement hiện tại densify TF‑IDF (`toarray()`) để tính cosine nhanh bằng `numpy.dot`. Với tập dữ liệu lớn hơn hoặc `max_features` lớn, có thể tốn RAM.

## Cấu trúc mã nguồn

- `knn_weight/data.py`: tải AG News + tách train/val/test
- `knn_weight/vectorize.py`: TF‑IDF features
- `knn_weight/models.py`: hai biến thể KNN (majority vote và weighted cosine)
- `knn_weight/experiment.py`: pipeline chạy thử và in accuracy
