# HƯỚNG DẪN THỰC THI LAB 21 TRÊN GOOGLE COLAB (T4 GPU)

> Tệp notebook chính: `colab/Lab21_RUN_ALL.ipynb`  
> Thời gian thực hiện: ~20 phút (chế độ test rút gọn) hoặc ~80–100 phút (chế độ nộp bài đầy đủ)

---

## 🚀 BƯỚC 1: KHỞI TẠO MÔI TRƯỜNG COLAB

1. Mở notebook qua liên kết GitHub chính thức:
   👉 **[Mở Lab21_RUN_ALL.ipynb trên Google Colab](https://colab.research.google.com/github/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)**
2. Bật GPU Tesla T4:
   - Trên thanh menu của Colab, chọn: **Runtime** $\rightarrow$ **Change runtime type**.
   - Mục **Hardware accelerator**: chọn **T4 GPU**.
   - Bấm **Save**.
3. > [!IMPORTANT]
   > **Cảnh báo Vlearn:** Mỗi khi có cập nhật mới từ repo GitHub, hãy **tải lại tab trình duyệt (F5/Reload)**, không chỉ bấm "Reconnect". Colab chỉ nạp mã từ GitHub 1 lần duy nhất lúc mở tab.

---

## 🏃 BƯỚC 2: CHẠY LẦN LƯỢT 4 Ô (CELLS) TRÊN NOTEBOOK

### 🔹 Ô 1: Setup — Clone Repo & Cài đặt Thư viện (~1 phút)
- Chạy cell 1. 
- Output kiểm tra:
  - Nhận diện đúng GPU: `Tesla T4`.
  - VRAM: $\approx 15.0\text{ GB}$.
  - Commit hash hiển thị rõ ràng.

### 🔹 Ô 2: Smoke Test — Kiểm tra Unit Tests & Data (~30 giây)
- Chạy cell 2: lệnh `!python scripts/verify.py --smoke`.
- Tiêu chí: Tất cả test suite chuyển màu xanh `[ ok ] unit tests`. Không tải trọng số, không tốn VRAM.

### 🔹 Ô 3: Chạy Pipeline NB1 $\rightarrow$ NB5 (~80 phút)
Tại cell 3 có giao diện tham số Form:

```python
COMPUTE_TIER = "T4"
EVAL_LIMIT   = ""      # ĐỂ TRỐNG KHI CHẠY NỘP BÀI! (Nếu test nhanh thì để "8")
STAGES       = "nb1 nb2 nb3 nb4 nb5"
```

#### Chiến lược chạy khuyến nghị:
1. **Lần 1 (Kiểm tra quy trình - ~17 phút)**:
   - Đặt `EVAL_LIMIT = "8"` rồi chạy. Script sẽ hoàn tất từ NB1 đến NB5 rất nhanh để bạn nắm được các bước và xác nhận mã nguồn không bị lỗi.
2. **Lần 2 (Chạy chính thức để lấy điểm nộp bài - ~80–100 phút)**:
   - Đặt **`EVAL_LIMIT = ""` (xóa sạch, để chuỗi rỗng)**.
   - Bấm chạy. Toàn bộ 50 mẫu eval sẽ được đánh giá đầy đủ.

> [!TIP]
> **Xử lý khi mạng đứt giữa chừng (Resume)**:
> Nếu phiên Colab bị gián đoạn ở NB4 hoặc NB5, bạn không cần chạy lại từ NB1!
> - Đổi `STAGES = "nb4 nb5"` (hoặc `STAGES = "nb5"`).
> - Các adapter đã lưu trong `adapters/` sẽ tự động được tái sử dụng.

### 🔹 Ô 4: Chạy Gatekeeper & Xem Kết Quả (~10 giây)
- Chạy cell 4 để nghiệm thu:
  - `!python scripts/verify.py`
  - In nội dung `results/runs.csv` và `results/verdict.json`.

---

## 📝 BƯỚC 3: ĐIỀN BÁO CÁO & PHẢN TƯ TRÊN COLAB

Bạn có thể chỉnh sửa trực tiếp 2 tệp báo cáo ngay trên giao diện file của Colab (bấm vào biểu tượng thư mục ở thanh bên trái):
1. `Day21-Track3-Finetuning-Lab/submission/REPORT.md`:
   - Điền thông tin Họ tên, MSSV, Tier `T4`, Model `unsloth/Qwen3.5-4B`.
   - Lấy số liệu từ `results/runs.csv` và `results/verdict.json` dán vào các bảng.
   - Lấy các ca định tính từ `results/qualitative.json` (nhớ chọn $\ge 2$ ca FT thua).
   - Viết phần kết luận $\ge 150$ từ.
   - **Xóa sạch 100% các ký tự `<điền>`, `<paste>`, `<0.xx>`**.
2. `Day21-Track3-Finetuning-Lab/submission/REFLECTION.md`:
   - Trả lời 5 câu hỏi tự phản tư.

Sau khi sửa xong, tại 1 ô code trống trong Colab, chạy lại kiểm tra:
```python
!python scripts/verify.py
```
Khi màn hình báo **`Ready to submit.`** (0 FAIL), bạn đã hoàn thành bài!

---

## 📦 BƯỚC 4: TẢI SẢN PHẨM VỀ MÁY TÍNH

Tạo thêm một ô code mới ở cuối notebook Colab và dán đoạn mã sau để tự động nén và tải về máy tính gói nộp bài chuẩn (Option A):

```python
# @title Đóng gói & Tải về máy tính
import subprocess
from google.colab import files

# Đóng gói kết quả và báo cáo
!zip -r lab21_submission.zip \
    submission/REPORT.md \
    submission/REFLECTION.md \
    results/ \
    adapters/correct/adapter_config.json \
    adapters/correct/adapter_model.safetensors \
    notebooks/

print("Đang tải file lab21_submission.zip về máy tính...")
files.download("lab21_submission.zip")
```

---

## ⚠️ NHỮNG ĐIỀU CẦN TRÁNH TRÊN COLAB
1. **Không để màn hình ngủ/tắt máy khi đang chạy Ô 3**: Quá trình train kéo dài hơn 1 tiếng, nếu Colab mất kết nối quá lâu phiên làm việc sẽ bị hủy.
2. **Không nộp bài khi còn `EVAL_LIMIT=8`**: Nếu quên để trống `EVAL_LIMIT`, `make verify` sẽ báo lỗi `FAIL: full eval set used`.
3. **Không xếp hạng các run bằng Train Loss**: Luôn xem điểm ở cột `target` trong bảng in ra ở Ô 4 hoặc tệp `results/autopsy.json`.
