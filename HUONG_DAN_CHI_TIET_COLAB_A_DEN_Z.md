# 📘 HƯỚNG DẪN THỰC HIỆN CHI TIẾT LAB 21 TRÊN GOOGLE COLAB (TỪ A ĐẾN Z)

> **Môn học**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning LLMs (LoRA, QLoRA, Mask Proof, Baselines & Regression Gate)  
> **Môi trường**: Google Colab Free (Tesla T4 GPU 16GB)  
> **Thời gian thực hiện**: ~20 phút (chế độ kiểm thử nhanh) hoặc ~80–100 phút (chế độ nộp bài chính thức)  
> **Mục tiêu**: Đạt trọn vẹn 100 điểm Rubric + tối đa 15 điểm thưởng Bonus.

---

## 📑 MỤC LỤC
1. [Bản chất bài toán & 2 câu hỏi quyết định điểm số](#1-bản-chất-bài-toán--2-câu-hỏi-quyết-định-điểm-số)
2. [Chuẩn bị môi trường Google Colab](#2-chuẩn-bị-môi-trường-google-colab)
3. [Quy trình thực thi 4 Ô (Cells) trên Colab](#3-quy-trình-thực-thi-4-ô-cells-trên-colab)
4. [Chi tiết từng Checkpoint (CP1 $\rightarrow$ CP5) bên trong pipeline](#4-chi-tiết-từng-checkpoint-cp1--cp5-bên-trong-pipeline)
5. [Hướng dẫn hoàn thiện Báo cáo (REPORT.md & REFLECTION.md)](#5-hướng-dẫn-hoàn-thiện-báo-cáo-reportmd--reflectionmd)
6. [Đóng gói nén ZIP & Tải về máy tính nộp bài](#6-đóng-gói-nén-zip--tải-về-máy-tính-nộp-bài)
7. [Các cạm bẫy dễ mất điểm & Cách phòng tránh](#7-các-cạm-bẫy-dễ-mất-điểm--cách-phòng-tránh)

---

## 🎯 1. BẢN CHẤT BÀI TOÁN & 2 CÂU HỎI QUYẾT ĐỊNH ĐIỂM SỐ

- **Bài toán thực nghiệm**: Phân loại và trích xuất thông tin từ vé hỗ trợ khách hàng (Customer Service Ticket) tiếng Việt sang cấu trúc JSON gồm 4 trường dữ liệu khách quan:
  `{"intent": "...", "urgency": "...", "product": "...", "sentiment": "..."}`.
- **Hai câu hỏi lab bắt buộc bạn phải trả lời và chứng minh**:
  1. **Phần được tính loss có đúng là câu trả lời không?** *(Chứng minh ở NB1 bằng giải mã ngược token - loss mask proof)*.
  2. **Bản fine-tune có thắng base model khi đã prompt tử tế (b) không — và bạn có phát hiện & giải trình được nếu nó không thắng?** *(Đo mốc ở NB2, phán quyết ở NB5)*.

> [!NOTE]
> **Triết lý chấm điểm của Lab 21:**  
> Điểm không nằm ở chỗ mô hình fine-tune của bạn bắt buộc phải thắng. Điểm nằm ở chỗ **bạn biết và chứng minh được bằng khoa học thực nghiệm** nó có thắng hay không. Một phán quyết `FAILED` được giải trình sâu sắc, logic vẫn nhận trọn vẹn điểm tối đa.

---

## 🛠️ 2. CHUẨN BỊ MÔI TRƯỜNG GOOGLE COLAB

1. **Mở tệp notebook tích hợp**:  
   👉 Truy cập liên kết: **[Lab21_RUN_ALL.ipynb trên Google Colab](https://colab.research.google.com/github/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)**
2. **Kích hoạt phần cứng GPU Tesla T4**:
   - Trên thanh menu của Colab: chọn **Runtime** $\rightarrow$ **Change runtime type**.
   - Tại mục **Hardware accelerator**: chọn **T4 GPU**.
   - Bấm **Save**.
3. > [!IMPORTANT]
   > **Cảnh báo sống còn từ Vlearn:**  
   > Mỗi khi repo GitHub có cập nhật, bạn phải **Reload (F5) lại tab trình duyệt**, không được chỉ bấm "Reconnect". Colab chỉ nạp mã notebook từ GitHub đúng 1 lần khi mở tab. Nếu không F5, Colab sẽ chạy code cũ và nổ lỗi bên trong `get_peft_model()`.

---

## 🚀 3. QUY TRÌNH THỰC THI 4 Ô (CELLS) TRÊN COLAB

Notebook `Lab21_RUN_ALL.ipynb` được thiết kế tự động hóa toàn bộ quá trình qua 4 ô lệnh từ trên xuống dưới:

```
┌─────────────────────────────────────────────────────────────────┐
│ Ô 1: Setup (Clone repo, cài đặt thư viện)              ~1 phút  │
├─────────────────────────────────────────────────────────────────┤
│ Ô 2: Smoke test (Kiểm tra unit tests, không tốn GPU)   ~30 giây │
├─────────────────────────────────────────────────────────────────┤
│ Ô 3: Core pipeline (Chạy tuần tự từ NB1 đến NB5)       ~80 phút │
├─────────────────────────────────────────────────────────────────┤
│ Ô 4: Gatekeeper (Kiểm tra tự động trước khi nộp bài)   ~10 giây │
└─────────────────────────────────────────────────────────────────┘
```

---

### 🔹 Ô 1: Setup — Clone Repo & Cài đặt Môi trường (~1 phút)
- **Hành động**: Nhấp vào nút Run ($\blacktriangleright$) ở Ô 1.
- **Mã thực thi**:
  - Clone repo `Day21-Track3-Finetuning-Lab`.
  - Cài đặt các gói từ `requirements.txt`.
- **Dấu hiệu thành công**: Màn hình in ra:
  ```
  commit : <mã hash>
  GPU    : Tesla T4
  VRAM   : 15.0 GB
  ```

---

### 🔹 Ô 2: Smoke Test — Kiểm tra Bộ Test Tự Động (~30 giây)
- **Hành động**: Chạy Ô 2 (`!python scripts/verify.py --smoke`).
- **Mục đích**: Kiểm tra import thư viện, dữ liệu hạt giống và chạy toàn bộ unit tests (`tests/`).
- **Dấu hiệu thành công**: Toàn bộ các dòng kiểm thử đều hiển thị `[  ok  ]`, kết thúc bằng:
  ```
  x passed · 0 warnings · 0 failures
  ```

---

### 🔹 Ô 3: Core Pipeline — Chạy Tự Động NB1 $\rightarrow$ NB5 (~80–100 phút)
Ô 3 có giao diện tham số Form (Colab UI) gồm 3 biến:
- `COMPUTE_TIER`: Chọn `"T4"` *(mặc định)*.
- `EVAL_LIMIT`: Giới hạn số lượng mẫu đánh giá.
- `STAGES`: Danh sách notebook cần chạy: `"nb1 nb2 nb3 nb4 nb5"`.

#### Chiến thuật thực thi khuyến nghị:

#### ⚡ Lượt 1: Chạy thử nghiệm nhanh (Smoke Pass - ~17 phút)
- **Cài đặt Form**:
  - `EVAL_LIMIT`: chọn `"8"`
- **Mục đích**: Chạy lướt qua toàn bộ pipeline để đảm bảo GPU hoạt động bình thường, không phát sinh lỗi nạp model, và bạn hiểu được luồng dữ liệu.

#### 🎯 Lượt 2: Chạy chính thức để lấy kết quả nộp bài (~80–100 phút)
- **Cài đặt Form**:
  - `EVAL_LIMIT`: **Xóa sạch, để chuỗi rỗng `""`**
- **Bấm chạy Ô 3**: Pipeline sẽ chạy đầy đủ 50 mẫu đánh giá trên tập target.

> [!TIP]
> **Xử lý sự cố mạng bị đứt giữa chừng (Resume)**:
> Nếu phiên Colab bị ngắt kết nối trong lúc đang chạy NB4 hoặc NB5, bạn **không cần chạy lại từ đầu**:
> 1. Sửa ô Form: `STAGES = "nb4 nb5"` (hoặc `STAGES = "nb5"`).
> 2. Các adapter đã lưu trước đó trong thư mục `adapters/` sẽ tự động được nhận diện và bỏ qua không phải train lại.

---

### 🔹 Ô 4: Gatekeeper — Kiểm Duyệt Tự Động & Xem Kết Quả (~10 giây)
- **Hành động**: Chạy Ô 4 (`!python scripts/verify.py`).
- **Nội dung kiểm tra tự động**:
  - `mask_proof.json`: Đủ 2 assert xanh và `supervised_fraction < 0.95`.
  - `baselines_frozen.json`: Tập eval đầy đủ (không dính cờ `smoke_mode`), $\text{target}(b) > \text{target}(a)$.
  - `runs.csv`: Đủ 4 run (`correct`, `attn_only`, `wrong_lr`, `qlora`), cùng số `max_steps`, tham số của `attn_only` lệch $< 5\%$.
  - `verdict.json`: Có phán quyết `PASSED` hoặc `FAILED`.
- **Xem nhanh số liệu**: Ô 4 tự động in ra bảng số liệu từ `runs.csv` và `verdict.json`.

---

## 🔍 4. CHI TIẾT TỪNG CHECKPOINT (CP1 $\rightarrow$ CP5) BÊN TRONG PIPELINE

Khi Ô 3 chạy, hệ thống sẽ lần lượt thực thi 5 Checkpoints kỹ thuật sau:

### 📍 CP1 (NB1): Dữ Liệu, Chat Template & Mask Proof
- **Làm gì**:
  - Đọc 250 mẫu CSKH, chia train/val bằng seed 42.
  - Đo phân phối token, xác định $p95$ để gán cho `max_length`.
  - Kiểm tra xem template có chèn khối `<think>` hay không (`results/template_check.json`).
  - Giải mã ngược nhãn `label != -100` để khẳng định: **Chỉ có câu trả lời mới được tính loss, câu hỏi bị che hoàn toàn**.
- **Tiêu chí thành công**:
  - `results/mask_proof.json`: `answer_is_supervised: true`, `question_is_masked: true`.
  - `supervised_fraction < 0.95` (thường khoảng 20%–40%).

---

### 📍 CP2 (NB2): Đo & Đóng Băng Ba Baseline Trước Khi Train
- **Làm gì**: Đo 2 mốc chuẩn của Base Model trước khi can thiệp trọng số:
  - **(a) Naive prompt**: Prompt thô sơ.
  - **(b) Optimized prompt**: Prompt có cấu trúc chuẩn, few-shot và ràng buộc schema JSON.
- **Tiêu chí thành công**:
  - `results/baselines_frozen.json` ghi nhận $\text{target}(b) > \text{target}(a)$.
  - Mã băm SHA256 của prompt (b) được khóa lại để bảo đảm liêm chính khoa học.

---

### 📍 CP3 (NB3): Huấn Luyện LoRA Chuẩn ("Vùng Không Hối Tiếc")
- **Làm gì**: Huấn luyện adapter `correct` với cấu hình chuẩn mực:
  - `target_modules`: Gắn vào toàn bộ linear của text decoder (`text-linear`).
  - `learning_rate`: $1 \times 10^{-4}$ (chuẩn LoRA, không dùng $10^{-5}$).
  - `effective_batch_size`: $< 32$.
  - $r = 16, \alpha = 32$.
- **Tiêu chí thành công**:
  - Adapter được lưu vào `adapters/correct/`.
  - Dòng `correct` được ghi vào `results/runs.csv`.

---

### 📍 CP4 (NB4): Mổ Xẻ 3 Cấu Hình Sai Đối Chứng Công Bằng
- **Làm gì**: Huấn luyện 3 run đối chứng với **cùng số bước `max_steps`**:
  1. `attn_only`: Chỉ gắn LoRA vào $q, v$, nhưng nâng rank lên ($r \approx 283$) để số tham số trainable bằng với `correct` ($\text{sai lệch} < 5\%$).
  2. `wrong_lr`: Dùng LR của Full-FT ($1 \times 10^{-5}$) $\rightarrow$ thấy loss gần như đi ngang.
  3. `qlora`: Nạp model dạng 4-bit NF4 $\rightarrow$ đo mức giảm VRAM (~41%).
- **Tiêu chí thành công**:
  - `results/runs.csv` có đủ 4 dòng với cùng `max_steps`.

---

### 📍 CP5 (NB5): Đánh Giá 4 Nhóm Chỉ Số & Phán Quyết
- **Làm gì**:
  - Đánh giá trên 4 nhóm: **Target** (độ chính xác JSON 4 trường), **Regression** (15 câu phổ thông kiểm tra quên thảm họa), **Format** (JSON hợp lệ), **Latency** (ms/mẫu).
  - Ra phán quyết Cổng hồi quy: `results/verdict.json`.
  - Đánh giá 3 run của NB4 trên tập Target $\rightarrow$ `results/autopsy.json`.
  - Trích xuất 5 ca so sánh định tính $\rightarrow$ `results/qualitative.json`.
- **Tiêu chí thành công**:
  - Xếp hạng 4 run bằng điểm **Target ở NB5**, không xếp bằng train loss ở NB4.
  - Có sẵn danh sách mẫu so sánh (trong đó có $\ge 2$ ca fine-tune thua).

---

## 📝 5. HƯỚNG DẪN HOÀN THIỆN BÁO CÁO (REPORT.md & REFLECTION.md)

Bạn mở thanh điều hướng tệp ở bên trái Colab, tìm đến thư mục `Day21-Track3-Finetuning-Lab/submission/`:

### 5.1. Hoàn thiện tệp `submission/REPORT.md`
1. **Thông tin đầu trang**:
   - Điền Họ tên, MSSV, Ngày làm bài.
   - Tier: `T4`, Base model: `unsloth/Qwen3.5-4B`, GPU thực tế: `Tesla T4 16GB`.
2. **Mục 1 — Setup**:
   - Điền số mẫu train/val (225 / 25).
   - `max_length`: Điền số đo $p95$ từ `results/token_stats.json`.
   - Template có khối `<think>` không: Trả lời theo `results/template_check.json`.
3. **Mục 2 — Mask Proof**:
   - Điền số `supervised_fraction` (ví dụ `0.32`).
   - Dán 3–5 dòng text đầu tiên được tính loss từ `results/mask_proof.json`.
4. **Mục 3 — Ba Baseline**:
   - Điền bảng điểm (a), (b), (c) từ `results/baselines_frozen.json` và `results/verdict.json`.
   - Trả lời xác nhận (b) có mạnh hơn (a) không.
5. **Mục 4 — Giải phẫu Cấu hình Sai**:
   - Điền bảng 4 dòng với điểm **target (NB5 §4)** lấy từ `results/autopsy.json`.
   - Trả lời 3 câu hỏi phân tích (4.1, 4.2, 4.3): Mỗi câu tối thiểu 3 câu văn có lập luận nguyên nhân - kết quả.
6. **Mục 5 — Phán Quyết (Verdict)**:
   - Ghi rõ `PASSED` hoặc `FAILED`, giá trị $\Delta_{\text{target}}$ và $\Delta_{\text{regression}}$.
   - Viết đoạn phân tích $\ge 100$ từ giải thích tại sao đạt kết quả đó.
7. **Mục 6 — Bảng Định Tính**:
   - Mở `results/qualitative.json`, chọn 5 trường hợp đưa vào bảng.
   - **Bắt buộc**: Phải có **ít nhất 2 ca Fine-tune THUA Baseline (b)**. Nêu nhận xét vì sao model thua ở các ca này.
8. **Mục 7 — Kết Luận & Bài Học**:
   - Viết kết luận $\ge 150$ từ về việc có nên triển khai thực tế bản fine-tune hay không.
   - Nêu 3 điều tâm đắc học được (cụ thể, thực tế).

> [!CAUTION]
> **Xóa sạch Placeholder**:  
> Kiểm tra lại toàn bộ bài viết, đảm bảo không còn bất kỳ ký tự `<điền>`, `<paste>`, `<0.xx>` nào sót lại trong tệp `REPORT.md`.

### 5.2. Hoàn thiện tệp `submission/REFLECTION.md`
Mở tệp và trả lời ngắn gọn, thẳng thắn 5 câu hỏi tự phản tư cá nhân.

### 5.3. Xác thực nghiệm thu bài làm
Tại 1 ô code trống trong Colab, chạy lại lệnh:
```python
!python scripts/verify.py
```
Nếu màn hình thông báo:
```
Ready to submit.
0 failures
```
Chúc mừng! Bạn đã hoàn thành bài lab một cách hoàn hảo.

---

## 📦 6. ĐÓNG GÓI NÉN ZIP & TẢI VỀ MÁY TÍNH NỘP BÀI

Tạo một ô code mới ở cuối notebook Colab, dán đoạn mã sau và bấm chạy:

```python
# @title Đóng gói sản phẩm nộp bài theo Option A (Chuẩn Rubric)
from google.colab import files

!zip -r lab21_submission.zip \
    submission/REPORT.md \
    submission/REFLECTION.md \
    results/ \
    adapters/correct/adapter_config.json \
    adapters/correct/adapter_model.safetensors \
    notebooks/

print("Đang tải file nén lab21_submission.zip về máy tính của bạn...")
files.download("lab21_submission.zip")
```

Tệp ZIP tải về máy tính sẽ có dung lượng khoảng ~10–15 MB, chứa đầy đủ các minh chứng thực nghiệm để nộp lên cổng học tập.

---

## ⚠️ 7. CÁC CẠM BẪY DỄ MẤT ĐIỂM & CÁCH PHÒNG TRÁNH

| Cạm bẫy | Hậu quả | Cách phòng tránh |
|---|---|---|
| **Chạy với `EVAL_LIMIT=8` khi nộp** | `verify.py` báo FAIL: `full eval set used` | Đặt `EVAL_LIMIT = ""` ở Ô 3 trước khi chạy lượt cuối. |
| **Xếp hạng bằng Train Loss ở NB4** | Mất điểm mục 2.5 và mục 3 | Luôn xếp hạng 4 cấu hình bằng cột **Target ở NB5** (`autopsy.json`). |
| **Chỉ báo cáo ca thắng ở mục Định tính** | Bị trừ sạch 5 điểm mục 3.4 (Cherry-pick) | Bắt buộc phải đưa và phân tích **$\ge 2$ ca Fine-tune THUA** baseline (b). |
| **Còn sót ký tự giữ chỗ trong REPORT** | `verify.py` báo FAIL: `REPORT.md filled in` | Dùng chức năng tìm kiếm (Ctrl+F) tìm `<` để xóa hết `<điền>`, `<paste>`. |
| **Báo cáo quá ngắn** | Bị cảnh báo Warn hoặc trừ điểm mục 4 | Đảm bảo báo cáo $\ge 400$ từ, kết luận $\ge 150$ từ, diễn giải phán quyết $\ge 100$ từ. |
