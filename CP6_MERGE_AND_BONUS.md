# CHECKPOINT 6 (CP6) — THỬ THÁCH NÂNG CAO & MERGE/SERVE (BONUS TỐI ĐA +15 ĐIỂM)

> **Notebook tương ứng**: `notebooks/06_merge_and_serve.py` & `BONUS-CHALLENGE.md`  
> **Thời gian thực thi**: ~10 phút (cho NB6) + thời gian thực hiện từng thử thách tự chọn  
> **Điểm Thưởng Rubric**: Tối đa **+15 điểm** cộng thêm vào tổng điểm lab

---

## 🎯 1. MỤC TIÊU CỦA CP6
1. **Thực hành triển khai mô hình trong thực tế**: Hợp nhất vĩnh viễn adapter LoRA vào trọng số gốc (Zero-overhead serving) và kỹ thuật hoán đổi adapter nóng (Hot-swapping) trên cùng một base model.
2. **Khám phá các hiện tượng tiên tiến (2026)**:
   - Thử nghiệm sụp đổ chuỗi suy luận (Reasoning-trace collapse).
   - Kiểm chứng thực nghiệm đòn bẩy của Rank ($r$) so với Vị trí và Learning Rate.
   - Ứng dụng trên tập dữ liệu nghiệp vụ tùy biến (Domain Dataset).
   - Chia sẻ mô hình lên cộng đồng mã nguồn mở HuggingFace Hub.

---

## 🛠️ 2. CHI TIẾT CÁC THỬ THÁCH THƯỞNG (BONUS CHALLENGES)

### 🎁 B1: Merge Trọng Số & Hot-Swap Phục Vụ Nhiều Adapter (+3 điểm)
- **Tệp thực thi**: `notebooks/06_merge_and_serve.py` (hoặc lệnh `make nb6`).
- **Nội dung làm**:
  1. Hợp nhất trọng số adapter `correct` vào Base model thông qua `model.merge_and_unload()`.
  2. Đo đạc lại độ chính xác trên tập `target`: Đảm bảo độ chênh lệch điểm số $\le 0.01$ (bảo toàn năng lực sau merge).
  3. Trải nghiệm cơ chế Hot-swap: Nạp 2 adapter khác nhau trên cùng 1 instance base model đang chạy trên VRAM mà không cần reload base model.
- **Sản phẩm nghiệm thu**: `results/merge_check.json`.
- **Câu hỏi tư duy trong Report**: Merge triệt tiêu overhead suy luận, nhưng bạn mất đi sự linh hoạt nào? Khi nào nên giữ adapter rời?

---

### 🎁 B2: Huấn Luyện Với Dataset Miền Nghiệp Vụ Riêng (+3 điểm)
- **Nội dung làm**:
  1. Chuẩn bị $\ge 200$ mẫu dữ liệu thuộc miền nghiệp vụ riêng của bạn (y tế, pháp luật, tài chính, logistics, hỗ trợ kỹ thuật...).
  2. Tạo tệp tài liệu `data/CUSTOM_DATASET.md` giải trình:
     - Nguồn gốc & quy trình thu thập dữ liệu.
     - Phương pháp khử nhiễm (Data Decontamination): Đảm bảo tuyệt đối không rò rỉ mẫu eval vào train.
     - Lý do tại sao tập dữ liệu này mới về mặt phân phối so với những gì base model đã thấy.
- **Lưu ý**: Có tệp `data/CUSTOM_DATASET.md` thì Gatekeeper `make verify` sẽ cho phép checksum tập eval thay đổi mà không báo FAIL.

---

### 🎁 B3: Hiện Tượng Reasoning-Trace Collapse (+4 điểm ⭐ Thử thách khó nhất)
- **Bối cảnh lý thuyết (Deck §17.5)**: Khi fine-tune một mô hình reasoning (như Qwen3.5) bằng dữ liệu hỏi đáp đơn thuần, mô hình có thể bị triệt tiêu năng lực suy luận từng bước (khối `<think>` bị rỗng hoặc lỗi) dù accuracy ngắn hạn vẫn tăng.
- **Cách thực hiện**:
  1. Chạy thí nghiệm lần 1:
     ```bash
     MASK_MODE=assistant-only make nb3 && make nb5
     ```
     Ghi lại chỉ số `valid_trace_rate`.
  2. Chạy thí nghiệm lần 2:
     ```bash
     MASK_MODE=response-only make nb3 && make nb5
     ```
     Ghi lại chỉ số `valid_trace_rate`.
  3. Lập bảng so sánh trong Phụ lục `REPORT.md`:

| MASK_MODE | Target Score | valid_trace_rate | Regression Score |
|---|---|---|---|
| `assistant-only` | | | |
| `response-only` | | | |

- **Câu hỏi tư duy**: Target có tăng trong khi `valid_trace_rate` tụt dốc không? Nếu chỉ nhìn vào Perplexity hoặc Accuracy thì có nhận ra lỗi này không?

---

### 🎁 B4: Quét Rank Có Kiểm Soát (Controlled Rank Sweep) (+3 điểm)
- **Bối cảnh lý thuyết (Deck §11)**: Rank là dung lượng biểu diễn tương ứng với thông tin mới trong dữ liệu, không phải là nút vặn tăng chất lượng tùy ý.
- **Cách thực hiện**:
  1. Cố định vị trí gắn = `text-linear`, cố định LR $= 10^{-4}$ và cùng số steps.
  2. Quét 3 giá trị rank: $r \in \{8, 16, 64\}$.
  3. So sánh biên độ thay đổi của Target Accuracy khi đổi $r$ so với biên độ khi đổi vị trí (`attn_only`) và đổi LR (`wrong_lr`).
  4. Xếp hạng 3 nút vặn (Vị trí, LR, Rank) theo mức độ ảnh hưởng thực tế.

---

### 🎁 B5: Đóng Gói & Xuất Bản Lên HuggingFace Hub (+2 điểm)
- **Cách thực hiện**:
  1. Đăng nhập HuggingFace CLI (`huggingface-cli login`).
  2. Đẩy adapter lên Hub:
     ```python
     model.push_to_hub("<your-username>/lab21-qwen35-triage-vi")
     ```
  3. Đính kèm liên kết adapter công khai vào `submission/REPORT.md` và `LINKS.md`.

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

```bash
# Kiểm thử chức năng merge:
make nb6

# Chạy toàn bộ pipeline bao gồm cả NB6:
make pipeline-full
```

---

## 📝 4. CHECKLIST CHI TIẾT CP6

- [ ] **Thực hiện Thử thách B1 (NB6 - Khuyến nghị)**:
  - [ ] Chạy `make nb6` hoàn tất thành công.
  - [ ] Tệp `results/merge_check.json` tồn tại.
  - [ ] Điểm số sau merge chênh lệch không quá $0.01$ so với adapter rời.
- [ ] **Thực hiện Thử thách B2 (Nếu chọn)**:
  - [ ] Chuẩn bị đủ $\ge 200$ mẫu.
  - [ ] Tạo đầy đủ tệp `data/CUSTOM_DATASET.md`.
- [ ] **Thực hiện Thử thách B3 (Nếu chọn)**:
  - [ ] Chạy đủ 2 chế độ `MASK_MODE`.
  - [ ] Điền bảng so sánh `valid_trace_rate` vào phụ lục report.
- [ ] **Thực hiện Thử thách B4 (Nếu chọn)**:
  - [ ] Quét đủ $r \in \{8, 16, 64\}$.
  - [ ] Kết luận về đòn bẩy của Rank.
- [ ] **Thực hiện Thử thách B5 (Nếu chọn)**:
  - [ ] Push adapter lên HuggingFace Hub.
  - [ ] Thêm link vào mục Phụ lục của `REPORT.md`.
- [ ] **Tích chọn các mục đã hoàn thành** trong phần Phụ lục của `submission/REPORT.md`.
