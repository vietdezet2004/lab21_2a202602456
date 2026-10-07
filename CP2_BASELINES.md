# CHECKPOINT 2 (CP2) — ĐÓNG BĂNG TẬP EVAL & ĐO 3 MỐC BASELINE TRƯỚC KHI TRAIN

> **Notebook tương ứng**: `notebooks/02_baselines.py`  
> **Thời gian thực thi**: ~17–23 phút trên GPU T4 (chạy sinh suy luận 2 lượt trên tập eval)  
> **Điểm Rubric liên quan**: 15 điểm (Mục 3.1: 5đ, Mục 3.2: 10đ)

---

## 🎯 1. MỤC TIÊU CỦA CP2
1. **Thiết lập mốc chuẩn khách quan trước khi can thiệp trọng số**: Đóng băng tập đánh giá (eval set) và đo đạc năng lực vốn có của Base model trước khi tiến hành bất kỳ bước fine-tune nào.
2. **Đo 2 baseline căn bản**:
   - **Baseline (a)**: Base model + Naive prompt (Prompt thô, zero-shot cơ bản).
   - **Baseline (b)**: Base model + Optimized prompt (Prompt kỹ thuật cao, có chỉ dẫn định dạng rõ ràng, few-shot).
3. **Chứng minh Baseline (b) là một đối thủ thực thụ**: Xác minh `(b) > (a)` trên thước đo `target`. Nếu prompt kỹ càng không thắng được prompt sơ sài, bạn chưa có một baseline hợp lệ để so sánh với Fine-tuning.
4. **Bảo toàn tính liêm chính nghiên cứu**: Tính và khóa mã băm SHA256 của `OPTIMIZED_PROMPT` nhằm ngăn chặn việc cố tình làm yếu prompt để khiến bản fine-tune trông có vẻ vượt trội.

---

## 🛠️ 2. CP2 LÀM NHỮNG VIỆC GÌ?

### Bước 1: Khóa Tập Đánh Giá (Freeze Eval Sets)
- Xác thực checksum SHA256 của hai tệp đánh giá:
  + `data/eval_target.jsonl`: 50 ticket CSKH tiếng Việt dành riêng để đo độ chính xác trích xuất JSON.
  + `data/eval_regression.jsonl`: 15 câu hỏi kiến thức / chỉ dẫn phổ thông tiếng Việt.
- Đảm bảo tập eval hoàn toàn tách biệt với tập train (không nhiễm bẩn dữ liệu - Data Contamination).

### Bước 2: Đo Đạc Baseline (a) — Base Model + Naive Prompt
- Sử dụng mô hình gốc (chưa nạp adapter).
- Đưa câu hỏi vào với prompt thô sơ: yêu cầu trích xuất thông tin mà không kèm quy tắc cấu trúc chặt chẽ.
- Thực hiện greedy decode để đảm bảo tính tất định (deterministic).
- Ghi nhận 4 nhóm chỉ số: `target`, `regression`, `format`, `latency`.

### Bước 3: Đo Đạc Baseline (b) — Base Model + Optimized Prompt
- Đưa câu hỏi vào với `OPTIMIZED_PROMPT`: có cấu trúc rõ ràng, mô tả chi tiết 4 trường JSON (`intent`, `urgency`, `product`, `sentiment`), các ràng buộc giá trị hợp lệ.
- Thực hiện greedy decode.
- Ghi nhận 4 nhóm chỉ số.
- So sánh đối soát trực tiếp: Khẳng định rằng `target(b) > target(a)`.

### Bước 4: Lưu Trữ Snapshot Đóng Băng
- Xuất toàn bộ kết quả vào tệp `results/baselines_frozen.json` cùng với:
  + Điểm số chi tiết của (a) và (b).
  + Mã SHA256 của prompt tối ưu (`optimized_prompt_sha`).
  + Số lượng mẫu eval đã chạy (`n_target = 50`).

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

### 3.1. Lệnh thực thi
```bash
# Chạy notebook 02 qua make:
make nb2

# Hoặc chạy trực tiếp script:
python notebooks/02_baselines.py

# Chạy unit test kiểm tra hàm đánh giá:
pytest tests/test_evaluate.py -v
```

### 3.2. Tiêu chí kiểm tra thành công (Verification Criteria)
Mở tệp `results/baselines_frozen.json` và kiểm tra các tiêu chí bắt buộc:
1. **`smoke_mode`**: Bắt buộc là `false` (hoặc không có cờ rút gọn). Nếu chạy với `EVAL_LIMIT=8` để test nhanh thì khi nộp bài phải bỏ cờ này và chạy lại đủ 50 mẫu.
2. **Tính vượt trội của (b)**:
   $$\text{target}_{(b)} > \text{target}_{(a)}$$
   (Ví dụ thực tế đo đạc: $a \approx 0.000$, $b \approx 0.765$).
3. **Format JSON**: `format` của baseline (b) phải đạt mức cao (lý tưởng là $1.000$ - tức 100% parse được JSON hợp lệ).
4. **Mã băm SHA**: Trường `optimized_prompt_sha` khớp với hàm băm của chuỗi `OPTIMIZED_PROMPT` trong mã nguồn.

---

## ⚠️ 4. CẠM BẪY & LỖI THƯỜNG GẶP (GOTCHAS)

> [!CAUTION]
> **Lỗi Liêm chính: Cố tình làm yếu Prompt (b)**:
> Gatekeeper (`scripts/verify.py`) kiểm tra SHA của prompt tối ưu. Nếu sửa prompt này yếu đi để model Fine-tune dễ thắng hơn, hệ thống sẽ cảnh báo hoặc đánh trượt. Nếu bạn cải thiện prompt này mạnh hơn, hãy tự tin ghi nhận trong `REPORT.md`.

> [!WARNING]
> **Chạy chế độ rút gọn (EVAL_LIMIT) khi nộp bài**:
> Đặt `EVAL_LIMIT=8` rất hữu ích khi debug nhanh (~2 phút), nhưng tệp `baselines_frozen.json` sẽ bị ghi cờ `smoke_mode: true`. Lệnh `make verify` sẽ báo **FAIL: full eval set used**. Nhớ chạy lại bản đầy đủ trước khi chốt nộp.

---

## 📝 5. CHECKLIST CHI TIẾT CP2

- [ ] **Môi trường**: Runtime có GPU sẵn sàng (Colab T4 hoặc GPU cá nhân) với torch CUDA hoạt động.
- [ ] **Tập dữ liệu Eval**: `data/eval_target.jsonl` (50 dòng) và `data/eval_regression.jsonl` (15 dòng) nguyên vẹn.
- [ ] **Thực thi NB2**: Chạy `make nb2` thành công, không gặp lỗi Out-Of-Memory (OOM).
- [ ] **Xác thực kết quả trong `results/baselines_frozen.json`**:
  - [ ] Đủ cả hai khối `baseline_a` và `baseline_b`.
  - [ ] `baseline_b.target` lớn hơn `baseline_a.target`.
  - [ ] `baseline_b.format` đạt chuẩn format cao (tiệm cận hoặc bằng 1.0).
  - [ ] `n_target` bằng đúng số lượng mẫu eval đầy đủ (50 mẫu).
  - [ ] Không dính cờ `smoke_mode`.
- [ ] **Ghi nhận số liệu**: Copy các số đo của (a) và (b) vào Bảng mục 3 trong `submission/REPORT.md`.
