# KNN Weight (Text Classification)


Mục tiêu: thử nghiệm KNN cho bài toán phân loại văn bản theo hướng **cosine similarity** trên đặc trưng **TF‑IDF**, đồng thời so sánh:
- **KNN_Optimized**: bỏ phiếu đa số (majority vote) trên top‑k láng giềng.
- **KNN_WeightedCosine**: bỏ phiếu có trọng số (weighted vote) theo độ tương tự cosine.
## Tài nguyên

- Notebook Colab (tham khảo/nguồn ý tưởng): https://colab.research.google.com/drive/1mNqr5vSHXIeetL-HozczCnukJTvro27M?usp=sharing
## Ý tưởng & cải tiến chính

### 1) Biểu diễn văn bản bằng TF‑IDF

Sử dụng `TfidfVectorizer` với cấu hình giống notebook:
- `max_features=5000`
- `ngram_range=(1, 2)`
- `stop_words='english'`

### 2) KNN với cosine similarity (chuẩn hoá L2)

Thay vì Euclidean distance trực tiếp trên TF‑IDF, ta chuẩn hoá mỗi vector về độ dài 1 và dùng tích vô hướng để tính cosine:

$$\text{cosine}(x, y) = \frac{x \cdot y}{\|x\|\,\|y\|}$$

Trong code:
- Dữ liệu train được chuẩn hoá trước (`X_train_norm`).
- Với mỗi batch dữ liệu đầu vào, chuẩn hoá rồi tính ma trận similarity bằng `np.dot(X_norm, X_train_norm.T)`.

### 3) “Optimized” top‑k

Hai phiên bản chọn láng giềng khác nhau:
- `KNN_Optimized`: dùng `np.argsort(...)[..., -k:]` (đơn giản, dễ hiểu).
- `KNN_WeightedCosine`: dùng `np.argpartition(..., -k)` để lấy top‑k nhanh hơn (không cần sort toàn bộ).

### 4) Weighted voting theo cosine

Ở `KNN_WeightedCosine`, mỗi láng giềng đóng góp một trọng số bằng similarity.
Để ổn định hơn, code đang **clip trọng số về không âm**:

$$w = \max(\text{cosine}, 0)$$

Điểm của mỗi lớp là tổng trọng số các láng giềng thuộc lớp đó, dự đoán lấy lớp có điểm cao nhất.

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

## Dataset & thiết lập thực nghiệm

### IMDb
- Nguồn: HuggingFace `datasets` (`load_dataset("imdb")`)
- Label: `0 = negative`, `1 = positive`
- Mặc định chạy nhanh (theo notebook): `train_size=25000`, `test_size=3000`
- Train được shuffle rồi chọn subset; đồng thời tách thêm validation 20% (đang giữ để giống notebook, nhưng script hiện chỉ đánh giá trên test).

### AG News
- Nguồn: HuggingFace `datasets` (`load_dataset("ag_news")`)
- Label: `0=World, 1=Sports, 2=Business, 3=Sci/Tech`
- Mặc định chạy nhanh (theo notebook): `train_size=8000`, `test_size=600`

## Cài đặt

Yêu cầu: Python 3.10+

```bash
pip install -r requirements.txt
```

Tuỳ chọn (cài dạng editable để import package thuận tiện):

```bash
pip install -e .
```

## Chạy thực nghiệm (reproduce)

### IMDb

```bash
python scripts/run_imdb.py --train-size 25000 --test-size 3000 --k 5
```

### AG News

```bash
python scripts/run_agnews.py --train-size 8000 --test-size 600 --k 5
```

Gợi ý lưu log kết quả ra file:

```bash
python scripts/run_imdb.py --k 5 > results_imdb.txt
python scripts/run_agnews.py --k 5 > results_agnews.txt
```

## Kết quả thực nghiệm

Kết quả phụ thuộc vào:
- Subset size (train/test)
- Tham số `k`
- Môi trường chạy (phiên bản thư viện, máy tính)

Bạn có thể chạy các lệnh ở trên rồi điền vào bảng sau (accuracy trên test):

| Dataset | Train/Test | k | KNN acc | KNN_Weighted acc |
|---|---:|---:|---:|---:|
| IMDb | 25000 / 3000 | 5 | (điền) | (điền) |
| AG News | 8000 / 600 | 5 | (điền) | (điền) |

Nếu bạn muốn so sánh nhiều giá trị `k`, chỉ cần chạy lại với `--k 1`, `--k 3`, `--k 5`, `--k 7`, ...

## Cấu trúc thư mục

- `src/knn_weight/knn.py`: 2 mô hình `KNN_Optimized`, `KNN_WeightedCosine`
- `src/knn_weight/data.py`: load dataset (IMDb, AG News)
- `src/knn_weight/vectorize.py`: TF‑IDF vectorizer
- `scripts/run_imdb.py`, `scripts/run_agnews.py`: entrypoints chạy thực nghiệm

## Hạn chế / lưu ý

- Hiện tại code gọi `toarray()` trước khi tính cosine để giống notebook; nếu tăng `train_size` lớn có thể tốn RAM.
- Lần đầu chạy sẽ tải dataset từ HuggingFace `datasets`.
