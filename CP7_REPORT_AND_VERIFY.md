# CHECKPOINT 7 (CP7) — BÁO CÁO KỸ THUẬT, PHẢN TƯ & NGHIỆM THU GATEKEEPER

> **Tệp thực hiện**: `submission/REPORT.md`, `submission/REFLECTION.md`, `scripts/verify.py`  
> **Thời gian thực hiện**: ~30 phút hoàn thiện tài liệu + chạy cổng kiểm duyệt  
> **Điểm Rubric liên quan**: 20 điểm (Mục 4.1: 5đ, Mục 4.2: 5đ, Mục 4.3: 5đ, Mục 4.4: 5đ)

---

## 🎯 1. MỤC TIÊU CỦA CP7
1. **Hoàn thiện Báo cáo Đánh giá (`submission/REPORT.md`)**: Trình bày toàn diện, khoa học, logic với số liệu trung thực, khớp 100% với các tệp JSON/CSV trong thư mục `results/`.
2. **Hoàn thành Tự phản tư (`submission/REFLECTION.md`)**: Trả lời trung thực 5 câu hỏi cốt lõi về trải nghiệm thực hành và tư duy kỹ thuật.
3. **Vượt qua Cổng Kiểm Duyệt Tự Động (Gatekeeper)**: Chạy `make verify` đạt trạng thái `Ready to submit` (0 FAIL).
4. **Đóng gói sản phẩm nộp bài**: Chuẩn bị tệp nộp bài theo đúng quy cách của 1 trong 3 Option (A, B, hoặc C).

---

## 🛠️ 2. CP7 LÀM NHỮNG VIỆC GÌ?

### Bước 1: Điền Toàn Diện Báo Cáo Kỹ Thuật (`submission/REPORT.md`)
Báo cáo mẫu được cung cấp sẵn, bạn cần điền đầy đủ và xóa sạch mọi thẻ giữ chỗ (`<điền>`, `<paste>`, `<0.xx>`):
1. **Thông tin chung**: Họ tên, MSSV, Ngày, Tier phần cứng, Base model ID, Thông tin GPU thực tế.
2. **Mục 1 — Setup**:
   - Tên dataset, số lượng mẫu train/val (seed 42).
   - Giá trị $p95$ đo được từ `results/token_stats.json` $\rightarrow$ giá trị `max_length`.
   - Kết quả kiểm tra Chat Template từ `results/template_check.json` (có khối `<think>` không và cách xử lý).
3. **Mục 2 — Mask Proof**:
   - Giá trị `supervised_fraction`, xác nhận `answer_is_supervised: true`, `question_is_masked: true`.
   - Dán 3–5 dòng văn bản thô đầu tiên thực sự được tính loss.
4. **Mục 3 — Ba Baseline**:
   - Điền đầy đủ số đo 4 nhóm của (a) Naive, (b) Optimized, (c) Fine-tune.
   - Trả lời xác nhận (b) có mạnh hơn (a) không. Nếu có sửa prompt (b), giải trình vì sao.
5. **Mục 4 — Giải phẫu Cấu hình Sai**:
   - Điền bảng 4 dòng: `correct`, `attn_only`, `wrong_lr`, `qlora` kèm điểm `target` thực tế đo được ở NB5 §4.
   - Trả lời 3 câu hỏi bắt buộc (mỗi câu tối thiểu 3 câu văn có quan hệ nhân quả):
     + *4.1 — Vị trí vs Rank*: So sánh `attn_only` và `correct`.
     + *4.2 — Đòn bẩy Learning Rate*: Tác động khi giảm LR 10 lần ở `wrong_lr`.
     + *4.3 — Sự đánh đổi của QLoRA*: Tiết kiệm VRAM bao nhiêu %, đánh đổi về tốc độ/độ chính xác.
6. **Mục 5 — Phán Quyết (Verdict)**:
   - Trạng thái `PASSED` hoặc `FAILED`, giá trị $\Delta_{\text{target}}$, $\Delta_{\text{regression}}$, `valid_trace_rate`.
   - Viết đoạn diễn giải $\ge 100$ từ.
7. **Mục 6 — Bảng Định Tính (Bắt buộc ca thua)**:
   - Liệt kê 5 ca so sánh thực tế.
   - Bắt buộc có **ít nhất 2 ca Fine-tune THUA Baseline (b)**.
   - Nhận xét tìm ra quy luật chung khiến Fine-tune dự đoán sai.
8. **Mục 7 — Kết luận & Điều Học Được**:
   - Kết luận $\ge 150$ từ về việc có nên deploy hay không và đâu là đòn bẩy thực sự.
   - 3 điều học được cụ thể, sâu sắc.
   - Kế hoạch thử nghiệm tiếp nếu có thêm 2 giờ.
9. **Phụ lục**: Tích chọn và dẫn link các phần thưởng (B1–B5) đã thực hiện.

---

### Bước 2: Hoàn Thành Tự Phản Tư (`submission/REFLECTION.md`)
Trả lời ngắn gọn, thẳng thắn 5 câu hỏi:
1. Điều làm bạn ngạc nhiên nhất?
2. Mất nhiều thời gian nhất ở công đoạn nào? Có đúng dự đoán không?
3. Trước lab này bạn tin điều gì về Fine-tuning mà giờ không còn tin?
4. Dùng AI assistant vào việc gì? Chỗ nào nó sinh mã hoặc gợi ý sai?
5. Nếu ngày mai phải fine-tune cho khách hàng thật, bước đầu tiên bạn làm là gì?

---

### Bước 3: Chạy Gatekeeper Kiểm Tra Toàn Diện
Thực thi lệnh kiểm tra:
```bash
make verify
# Hoặc: python scripts/verify.py
```
Gatekeeper sẽ kiểm tra chéo:
- Tồn tại đủ tất cả các tệp trong `results/`.
- Không còn bất kỳ placeholder nào trong `REPORT.md` và độ dài $\ge 400$ từ.
- Mask proof pass cả 2 assert và `supervised_fraction < 0.95`.
- Tập eval đầy đủ (không dính `smoke_mode`).
- SHA của prompt (b) không bị làm yếu đi và $\text{target}(b) > \text{target}(a)$.
- Checksum tập eval không bị biến đổi bất thường.
- Cả 4 run huấn luyện có chung một ngân sách `max_steps`.
- Sai lệch tham số trainable của `attn_only` $< 5\%$.
- Tệp `verdict.json` có phán quyết.

---

### Bước 4: Lựa Chọn Định Dạng Nộp Bài

| Định dạng | Cấu trúc thư mục | Khi nào chọn |
|---|---|---|
| **Option A (ZIP gọn ~5–15 MB - Mặc định)** | `submission/REPORT.md`<br>`results/` (đầy đủ các file .json, .csv)<br>`adapters/correct/` (chỉ adapter chính)<br>`notebooks/` | Phổ biến nhất, đầy đủ minh chứng thực nghiệm |
| **Option B (GitHub + HuggingFace Hub)** | `submission/REPORT.md`<br>`results/`<br>`LINKS.md` (chứa URL repo + URL HF model) | Nhận thêm +2 điểm thưởng B5 |
| **Option C (Code-only ~500 KB)** | `submission/REPORT.md`<br>`results/`<br>`requirements.txt` | Mạng yếu hoặc giới hạn dung lượng tải lên |

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

```bash
# Kiểm tra khói nhanh:
make smoke

# Kiểm tra toàn diện trước khi nộp bài:
make verify
```

### Tiêu chí thành công:
Đầu ra terminal hiển thị dòng thông báo màu xanh:
```
Ready to submit.
```
Exit code $= 0$, không có bất kỳ dòng nào mang nhãn `[ FAIL ]`.

---

## 📝 4. CHECKLIST CHI TIẾT CP7

- [ ] **Báo cáo `submission/REPORT.md`**:
  - [ ] Đã điền Họ tên, MSSV, Ngày, Tier, Base model ID.
  - [ ] Đã xóa sạch 100% placeholder (`<điền>`, `<paste>`, `<0.xx>`).
  - [ ] Toàn bộ số liệu khớp tuyệt đối với `results/*.json` và `results/runs.csv`.
  - [ ] Mục 4 trả lời đủ 3 câu hỏi phân tích, mỗi câu $\ge 3$ câu văn.
  - [ ] Mục 5 có đoạn diễn giải phán quyết $\ge 100$ từ.
  - [ ] Mục 6 có đủ 5 ca định tính với ít nhất 2 ca Fine-tune THUA.
  - [ ] Mục 7 có kết luận $\ge 150$ từ và 3 bài học thực tiễn.
  - [ ] Tổng độ dài báo cáo $\ge 400$ từ.
- [ ] **Phản tư `submission/REFLECTION.md`**:
  - [ ] Trả lời đầy đủ cả 5 câu hỏi phản tư cá nhân.
- [ ] **Nghiệm thu Gatekeeper**:
  - [ ] Chạy `python scripts/verify.py` đạt **0 FAIL**.
  - [ ] Đọc và giải quyết các cảnh báo `[ warn ]` nếu có.
- [ ] **Đóng gói**:
  - [ ] Nén thư mục nộp bài theo đúng quy cách MSSV (ví dụ: `lab21_2a202602456.zip`).
  - [ ] Đảm bảo thư mục nén chứa đầy đủ thư mục `results/` và báo cáo.
