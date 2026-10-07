# CHECKPOINT 5 (CP5) — ĐÁNH GIÁ 4 NHÓM, CỔNG HỒI QUY & PHÁN QUYẾT

> **Notebook tương ứng**: `notebooks/05_evaluate_and_verdict.py`  
> **Thời gian thực thi**: ~21 phút trên GPU T4 (đánh giá 4 run trên tập eval)  
> **Điểm Rubric liên quan**: 25 điểm (Mục 3.1: 5đ, Mục 3.2: 10đ, Mục 3.3: 5đ, Mục 3.4: 5đ)

---

## 🎯 1. MỤC TIÊU CỦA CP5
1. **Khép lại bảng 3 Baseline**: Hoàn thành cột kết quả của `(c) LoRA fine-tune` và so sánh trực diện với `(a) Naive prompt` và `(b) Optimized prompt`.
2. **Kích hoạt Cổng Hồi Quy (Regression Gate)**: Kiểm tra đồng thời sự tiến bộ trên bài toán đích (`target`) và bảo toàn năng lực tổng quát (`regression`), không để xảy ra hiện tượng "quên thảm họa".
3. **Chấm điểm các cấu hình sai bằng thước đo chuẩn**: Đánh giá 3 run đối chứng của NB4 trên tập `target`, đập tan ngộ nhận từ `train_loss`.
4. **Phân tích định tính đa chiều**: Trích xuất tối thiểu 5 trường hợp cụ thể, bắt buộc phân tích **ít nhất 2 ca Fine-tune THUA Baseline (b)** nhằm loại trừ hành vi chọn lọc thiên vị (cherry-picking).

---

## 🛠️ 2. CP5 LÀM NHỮNG VIỆC GÌ?

### Bước 1: Đánh Giá Toàn Diện Bản Huấn Luyện Chuẩn (`correct`)
Mô hình sau khi nạp adapter `correct` được đánh giá trên 4 nhóm chỉ số:
1. **Target Accuracy**: Độ chính xác trích xuất JSON so với ground truth 50 mẫu.
2. **Regression Score**: Điểm số trên 15 câu hỏi kiến thức / chỉ dẫn thông thường.
3. **Format Score**: Tỷ lệ câu trả lời parse thành công ra JSON và có đủ 4 khóa.
4. **Latency (ms/sample)**: Thời gian trung bình để sinh 1 mẫu ở chế độ greedy decoding.

### Bước 2: Kích Hoạt Cổng Hồi Quy (Regression Gate) & Ra Phán Quyết
Hệ thống tính toán các độ lệch:
$$\Delta_{\text{target}} = \text{target}_{(c)} - \text{target}_{(b)}$$
$$\Delta_{\text{regression}} = \text{regression}_{(c)} - \text{regression}_{(b)}$$
- **Điều kiện PASS**:
  - $\Delta_{\text{target}} \ge 0$ (Fine-tune phải thắng hoặc hòa prompt tối ưu).
  - $\Delta_{\text{regression}} \ge -0.05$ (Không được làm tụt quá 5% điểm năng lực chung).
  - $\text{format}_{(c)} \ge 0.95$.
- Xuất kết quả vào tệp `results/verdict.json`.

### Bước 3: Đánh Giá 3 Run Đối Chứng Của NB4 Trên Bài Toán Đích
- Nạp lần lượt các adapter `attn_only`, `wrong_lr`, `qlora` và đo điểm `target`.
- Xuất kết quả vào tệp `results/autopsy.json`.
- Thiết lập bảng xếp hạng thực sự của 4 cấu hình:
  $$\text{Xếp hạng theo Target Accuracy vs Xếp hạng theo Train Loss}$$
- Chỉ ra hiện tượng đảo chiều (ví dụ: `attn_only` train loss thấp hơn nhưng target accuracy lại kém hơn `correct`).

### Bước 4: Trích Xuất & Phân Tích Mẫu Định Tính
- Tự động tìm kiếm các trường hợp sai lệch giữa ground truth, kết quả của prompt (b) và kết quả của fine-tune (c).
- Xuất danh sách vào tệp `results/qualitative.json`.
- Lọc ra tối thiểu 5 ca, trong đó bắt buộc có:
  - $\ge 2$ ca Fine-tune THẮNG Baseline (b).
  - $\ge 2$ ca Fine-tune THUA Baseline (b).

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

### 3.1. Lệnh thực thi
```bash
# Chạy đánh giá và phán quyết:
make nb5

# Hoặc:
python notebooks/05_evaluate_and_verdict.py
```

### 3.2. Tiêu chí kiểm tra thành công (Verification Criteria)
Kiểm tra sự tồn tại và nội dung của 3 tệp kết quả:
1. **`results/verdict.json`**:
   - Chứa trường `"verdict"` với các khóa: `passed` (true/false), `target_delta`, `regression_delta`.
2. **`results/autopsy.json`**:
   - Ghi nhận đầy đủ điểm số `target` cho cả 4 run: `correct`, `attn_only`, `wrong_lr`, `qlora`.
3. **`results/qualitative.json`**:
   - Chứa danh sách các mẫu định tính với đầy đủ trường dữ liệu so sánh.

---

## ⚠️ 4. CẠM BẪY & LỖI THƯỜNG GẶP (GOTCHAS)

> [!CAUTION]
> **Chỉ Báo Cáo Ca Thắng (Cherry-Picking - Bị Trừ Hết 5 Điểm)**:
> Rubric 3.4 nêu rõ: *"≥5 ví dụ định tính, trong đó ≥2 ca fine-tune THUA. Chỉ chọn ca thắng = cherry-pick, trừ hết mục này"*. Bắt buộc phải đưa và phân tích ít nhất 2 ca mô hình fine-tune dự đoán sai hoặc kém hơn prompt (b) vào Bảng mục 6 của báo cáo.

> [!NOTE]
> **Tâm Lý Sợ Phán Quyết FAILED**:
> Nếu $\text{target}_{(c)} < \text{target}_{(b)}$ và kết quả là `FAILED`, bạn **VẪN ĐẠT ĐIỂM TỐI ĐA** nếu phân tích nguyên nhân trung thực trong report (do dữ liệu 250 mẫu chưa đủ đa dạng, base model đã quá mạnh ở prompt engineering, hay cấu hình hyperparameter chưa tối ưu). Tuyệt đối không chỉnh sửa số liệu giả tạo để cố ép kết quả thành `PASSED`.

---

## 📝 5. CHECKLIST CHI TIẾT CP5

- [ ] **Thực thi NB5**: Chạy `make nb5` hoàn tất không phát sinh lỗi nạp adapter hay OOM.
- [ ] **Xác thực `results/verdict.json`**:
  - [ ] Đọc giá trị `target_delta`, `regression_delta`, `format_c`, `latency_c`.
  - [ ] Ghi nhận trạng thái `PASSED` hoặc `FAILED`.
- [ ] **Xác thực `results/autopsy.json`**:
  - [ ] Điền cột **target (NB5 §4)** vào Bảng mục 4 trong `submission/REPORT.md`.
  - [ ] So sánh thứ tự giữa cột `train loss` và cột `target`.
- [ ] **Xác thực `results/qualitative.json`**:
  - [ ] Chọn ra 5 mẫu tiêu biểu.
  - [ ] Đảm bảo có ít nhất 2 ca Fine-tune THẮNG.
  - [ ] Đảm bảo có ít nhất 2 ca Fine-tune THUA.
  - [ ] Điền vào Bảng mục 6 trong `submission/REPORT.md` kèm nhận xét nguyên nhân.
