# CHECKPOINT 1 (CP1) — CHUẨN BỊ DỮ LIỆU & CHỨNG MINH LOSS MASK

> **Notebook tương ứng**: `notebooks/01_data_and_mask.py`  
> **Thời gian thực thi**: ~25 giây (chạy hoàn toàn trên CPU, không cần GPU)  
> **Điểm Rubric liên quan**: 20 điểm (Mục 1.1: 10đ, Mục 1.2: 5đ, Mục 1.3: 5đ)

---

## 🎯 1. MỤC TIÊU CỦA CP1
1. **Chia dữ liệu chuẩn xác và đo đạc phân phối token**: Tính toán độ dài chuỗi tại bách phân vị thứ 95 ($p95$) để đặt `max_length` bằng số đo thực nghiệm, không đoán mò.
2. **Khám phá hành vi Chat Template**: Kiểm tra xem template có tự động chèn hoặc giữ khối `<think>` hay không (đặc biệt quan trọng với các mô hình 2026 như Qwen3.5).
3. **Chứng minh Loss Mask bằng giải mã ngược (Reverse Decode)**: Chứng minh toán học và logic rằng mô hình chỉ học dự đoán câu trả lời của trợ lý (assistant response), toàn bộ câu hỏi (user prompt + system prompt) bị che hoàn toàn (`label = -100`).

---

## 🛠️ 2. CP1 LÀM NHỮNG VIỆC GÌ?

### Bước 1: Khởi tạo Dữ liệu & Tách tập (Split)
- Tải tập dữ liệu gốc `data/train_seed.jsonl` (250 ticket CSKH tiếng Việt gắn nhãn JSON 4 trường: `intent`, `urgency`, `product`, `sentiment`).
- Chia tập theo tỉ lệ cố định với `seed = 42` (mặc định 225 train / 25 val).

### Bước 2: Thống kê Token & Xác định `max_length`
- Áp dụng Chat Template lên từng mẫu dữ liệu thông qua tokenizer của base model.
- Thu thập độ dài chuỗi token sau khi render.
- Tính toán các giá trị thống kê: min, mean, p50 (median), p95, max.
- Lưu lại cấu hình `max_length` khớp với giá trị $p95$ đo được.

### Bước 3: Kiểm tra Chat Template & Khối `<think>`
- Kiểm tra hành vi của `tokenizer.apply_chat_template` khi render một hội thoại thông thường.
- Xác định xem template có tự động sinh cấu trúc `<think>\n\n</think>\n\n` hay không.
- Ghi nhận trạng thái để tránh lỗi bất đồng bộ định dạng giữa huấn luyện và sinh suy luận (F-01, F-25).

### Bước 4: Tạo Mask và Chứng minh Bằng Giải mã Ngược
- Sinh nhãn `labels` từ `input_ids`, gán giá trị `-100` cho tất cả các token thuộc system và user prompt.
- Chỉ giữ lại token IDs thực sự của câu trả lời (bao gồm token kết thúc `<|im_end|>`).
- Thực hiện giải mã ngược (decode) các token có `label != -100` thành văn bản thô.
- Assert kiểm tra:
  + `answer_is_supervised`: Đoạn text decode được phải khớp với câu trả lời mục tiêu.
  + `question_is_masked`: Đoạn text của prompt không được xuất hiện trong phần tính loss.
  + `supervised_fraction`: Tỷ lệ token tính loss / tổng số token phải nằm trong khoảng hợp lý (thường 15% - 40%), tuyệt đối `< 0.95`.

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

### 3.1. Lệnh thực thi
```bash
# Chạy trực tiếp qua make:
make nb1

# Hoặc chạy script python:
python notebooks/01_data_and_mask.py

# Chạy unit tests riêng cho phần mask và data:
pytest tests/test_masking.py tests/test_repo_structure.py -v
```

### 3.2. Tiêu chí kiểm tra thành công (Verification Criteria)
Chạy xong, kiểm tra 3 tệp JSON được tạo ra trong thư mục `results/`:
1. **`results/mask_proof.json`**:
   - `answer_is_supervised`: phải là `true`.
   - `question_is_masked`: phải là `true`.
   - `supervised_fraction`: phải `< 0.95` (nếu $\ge 0.95$ sẽ bị Gatekeeper đánh rớt và mất 10 điểm mục 1.1).
   - Chứa đoạn trích xuất 3–5 dòng đầu của văn bản thực sự được tính loss.
2. **`results/template_check.json`**:
   - Tồn tại và ghi nhận rõ cờ `has_think_block` (true/false) cùng chuỗi phân tách assistant turn.
3. **`results/token_stats.json`**:
   - Chứa các thông số đo đạc cụ thể: `p95`, `recommended_max_length`, `max_seen`.

---

## ⚠️ 4. CẠM BẪY & LỖI THƯỜNG GẶP (GOTCHAS)

> [!CAUTION]
> **Lỗi Supervised Prompt (`supervised_fraction >= 0.95`)**:
> Nếu dùng mask mode sai (ví dụ `MASK_MODE=everything`), mô hình sẽ học thuộc cả câu hỏi của người dùng. Khi suy luận, model có xu hướng lặp lại câu hỏi thay vì trả lời. Đây là lỗi nghiêm trọng nhất.

> [!WARNING]
> **Lỗi Diff Token List trên Qwen3.5 (F-01)**:
> Không được diff danh sách token dạng `full_tokens[len(prefix_tokens):]` vì ký tự xuống dòng `\n` ở cuối prefix và `\n\n` ở đầu câu trả lời có thể bị tokenizer gộp thành 1 token hoàn toàn khác. Phải dùng `return_offsets_mapping=True` và map theo span ký tự.

---

## 📝 5. CHECKLIST CHI TIẾT CP1

- [ ] **Môi trường**: Đã tạo môi trường venv và cài đặt tối thiểu `requirements-cpu.txt`.
- [ ] **Dữ liệu**: Tệp `data/train_seed.jsonl` tồn tại đủ 250 mẫu.
- [ ] **Thực thi NB1**: Chạy `make nb1` thành công không phát sinh exception.
- [ ] **Xác thực Mask Proof**:
  - [ ] Mở `results/mask_proof.json`, kiểm tra `"answer_is_supervised": true`.
  - [ ] Kiểm tra `"question_is_masked": true`.
  - [ ] Kiểm tra `"supervised_fraction"` nằm trong khoảng an toàn (thường $0.20 - 0.40$).
- [ ] **Xác thực Token Stats**:
  - [ ] Mở `results/token_stats.json`, lấy giá trị `p95`.
  - [ ] Xác nhận `max_length` được thiết lập bao trọn giá trị $p95$.
- [ ] **Xác thực Chat Template**:
  - [ ] Mở `results/template_check.json`, kiểm tra xem template có giữ khối `<think>` hay không để chuẩn bị ghi vào mục 1 của `REPORT.md`.
- [ ] **Unit Tests**: Chạy `pytest tests/test_masking.py` và tất cả các test ca đều PASS (xanh).
