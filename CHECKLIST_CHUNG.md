# LAB 21 — CHECKLIST TỔNG QUAN & LỘ TRÌNH THỰC HIỆN

> **Chủ đề**: Fine-tuning LLMs (LoRA, QLoRA, Mask Proof, Regression Gate, Evaluation)  
> **Mã môn**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn  
> **Thang điểm**: 100 điểm chính + tối đa 15 điểm thưởng Bonus

---

## 🎯 1. TỔNG QUAN: MỤC ĐÍCH & YÊU CẦU CẦN ĐẠT CỦA LAB 21

### 1.1. Mục đích cốt lõi
1. **Làm chủ quy trình Fine-tuning LLM chuẩn mực (2026)**: Sử dụng kỹ thuật Parameter-Efficient Fine-Tuning (LoRA / QLoRA) với cấu hình "vùng không hối tiếc" (LoRA Without Regret: đủ lớp, đúng thang Learning Rate, effective batch size < 32).
2. **Chứng minh toán học và logic của Loss Mask**: Chứng minh không bằng "niềm tin" mà bằng giải mã ngược token (Reverse Decode) rằng loss chỉ được tính trên phần câu trả lời (assistant response), prompt được che (mask) hoàn toàn (`supervised_fraction < 0.95`).
3. **Thực hiện một phép so sánh khoa học và công bằng (Fair Comparison)**:
   - Đóng băng mốc đánh giá và đo **3 baseline trước khi train**: (a) Base + naive prompt, (b) Base + optimized prompt, và sau đó mới so với (c) Fine-tune.
   - Run đối chứng (`attn_only`) phải dùng rank khớp ngân sách tham số (`matched_rank()`, lệch < 5% so với run chuẩn) và cùng số bước huấn luyện (`max_steps`).
4. **Đánh giá đa chiều qua Cổng Hồi Quy (Regression Gate) 4 nhóm**:
   - `target`: Độ chính xác trích xuất JSON 4 trường (`intent`, `urgency`, `product`, `sentiment`).
   - `regression`: Đánh giá 15 câu hỏi phổ thông để phát hiện hiện tượng quên thảm hoạ (Catastrophic Forgetting).
   - `format`: Tỷ lệ JSON parse được và đủ 4 khóa.
   - `latency`: Tốc độ sinh (ms/sample).
5. **Tư duy phán quyết trung thực**: Điểm số nằm ở chỗ bạn **biết và chứng minh được** mô hình fine-tune có thắng hay không. Một phán quyết `FAILED` được giải thích tường tận, khoa học vẫn đạt điểm tối đa (thậm chí cao hơn `PASSED` mà cherry-pick).

---

## 📊 2. BẢNG TIÊU CHÍ ĐÁNH GIÁ (RUBRIC - 100 ĐIỂM + 15 THƯỞNG)

| Tiêu chí | Trọng số | Yêu cầu cốt lõi |
|---|:---:|---|
| **1. Tính đúng đắn của pipeline** | **30 điểm** | • `mask_proof.json` pass cả 2 assert (answer tính loss, question mask hoàn toàn)<br>• `template_check.json` xác nhận xử lý `<think>`<br>• `max_length` đặt theo p95 đo đạc thực tế<br>• `adapters/correct/` train thành công, log đầy đủ vào `runs.csv` |
| **2. Thiết kế thí nghiệm công bằng** | **25 điểm** | • `attn_only` khớp ngân sách tham số trainable (< 5% sai lệch)<br>• Cả 4 run chung đúng 1 ngân sách `max_steps`<br>• Mỗi run chỉ đổi đúng 1 biến duy nhất<br>• Phân tích sâu vị trí (`text-linear` vs `q,v`) vs rank<br>• Xếp hạng các run bằng **target accuracy ở NB5**, KHÔNG xếp bằng training loss ở NB4 |
| **3. Chất lượng đánh giá & phán quyết** | **25 điểm** | • Baseline (b) đo trước khi train và chứng minh được `(b) > (a)`<br>• Đủ 4 nhóm chỉ số: target, regression, format, latency<br>• `verdict.json` có phán quyết và report diễn giải logic<br>• ≥ 5 mẫu định tính, trong đó bắt buộc có **≥ 2 ca fine-tune THUA** |
| **4. Chất lượng báo cáo kỹ thuật** | **20 điểm** | • Cấu trúc báo cáo đầy đủ luận điểm, giải trình lựa chọn model/dataset<br>• Kết luận ≥ 150 từ có quan hệ nhân quả<br>• Số liệu trong report khớp 100% với `results/`<br>• Phản tư cá nhân sâu sắc, không chung chung |
| **🎁 Thưởng (Bonus)** | **+15 điểm** | • B1 (+3): NB6 Merge adapter + Hot-swap<br>• B2 (+3): Dataset miền riêng ≥ 200 mẫu kèm khử nhiễm<br>• B3 (+4): Thí nghiệm Reasoning-trace collapse<br>• B4 (+3): Quét rank có kiểm soát ($r \in \{8, 16, 64\}$)<br>• B5 (+2): Push adapter lên HuggingFace Hub |

---

## 🗺️ 3. LỘ TRÌNH 7 CHECKPOINTS (CP)

| Checkpoint | File chi tiết | Nội dung chính | Môi trường / Thời gian (T4) | Sản phẩm đầu ra |
|:---:|---|---|:---:|---|
| **CP1** | [CP1_DATA_AND_MASK.md](CP1_DATA_AND_MASK.md) | Data split, token stats (p95), Chat template check, Mask proof | CPU (~25 giây) | `mask_proof.json`, `template_check.json`, `token_stats.json` |
| **CP2** | [CP2_BASELINES.md](CP2_BASELINES.md) | Đóng băng tập eval, đo baseline (a) Naive & (b) Optimized trước train | GPU (~17–23 phút) | `results/baselines_frozen.json` |
| **CP3** | [CP3_TRAIN_CORRECT.md](CP3_TRAIN_CORRECT.md) | Huấn luyện LoRA chuẩn (`text-linear`, $r=16$, $LR=10^{-4}$, batch < 32) | GPU (~15–25 phút) | Thư mục `adapters/correct/`, dòng `correct` trong `runs.csv` |
| **CP4** | [CP4_MISCONFIG_AUTOPSY.md](CP4_MISCONFIG_AUTOPSY.md) | Chạy 3 run đối chứng: `attn_only` (matched rank), `wrong_lr`, `qlora` | GPU (~45–60 phút) | 3 dòng đối chứng trong `results/runs.csv` |
| **CP5** | [CP5_EVALUATE_AND_VERDICT.md](CP5_EVALUATE_AND_VERDICT.md) | Đánh giá 4 nhóm, Cổng hồi quy, xếp hạng đối chứng, 5 ca định tính | GPU (~21 phút) | `verdict.json`, `autopsy.json`, `qualitative.json` |
| **CP6** | [CP6_MERGE_AND_BONUS.md](CP6_MERGE_AND_BONUS.md) | NB6 Merge weights, Hot-swap adapter, Các thử thách Bonus B1–B5 | GPU (~10 phút + bonus) | `results/merge_check.json`, HF Hub link, B3/B4 logs |
| **CP7** | [CP7_REPORT_AND_VERIFY.md](CP7_REPORT_AND_VERIFY.md) | Viết Báo cáo `REPORT.md`, Phản tư `REFLECTION.md`, chạy `make verify` | CPU (~30 phút) | `submission/REPORT.md`, `REFLECTION.md`, Gatekeeper 0 FAIL |

---

## 📋 4. CHECKLIST TIẾN ĐỘ TỔNG QUAN

### Giai đoạn 1: Thiết lập & Kiểm tra khói (Smoke Test)
- [ ] Đã sao chép `.env.example` thành `.env` và chọn tier phần cứng thích hợp (`T4` hoặc `LAPTOP`/`BIGGPU`/`CPU`).
- [ ] Chạy `make smoke` (hoặc `python scripts/verify.py --smoke`) thành công (tất cả unit tests xanh).

### Giai đoạn 2: Thực thi tuần tự các Checkpoints
- [ ] **CP1 (NB1)**: Đã chứng minh Loss Mask (`supervised_fraction < 0.95`, question masked, answer supervised) và tính xong p95 `max_length`.
- [ ] **CP2 (NB2)**: Đã đo xong 2 baseline trước khi train, kiểm tra `(b) > (a)` và ghi nhận SHA prompt.
- [ ] **CP3 (NB3)**: Đã train hoàn tất cấu hình chuẩn `correct`, adapter lưu tại `adapters/correct/`.
- [ ] **CP4 (NB4)**: Đã train xong 3 run đối chứng (`attn_only`, `wrong_lr`, `qlora`) với cùng số step.
- [ ] **CP5 (NB5)**: Đã đánh giá 4 nhóm chỉ số, có phán quyết cổng hồi quy, xếp hạng đối chứng theo Target accuracy, lọc ra ít nhất 2 ca FT thua.
- [ ] **CP6 (NB6 + Bonus - Tùy chọn)**: Đã thực hiện ít nhất 1 bài toán thưởng (B1 merge / B2 dataset riêng / B3 reasoning collapse / B4 rank sweep / B5 HF Hub).

### Giai đoạn 3: Báo cáo & Kiểm duyệt nghiệm thu
- [ ] **CP7 (Report & Verify)**: Đã điền 100% dữ liệu vào `submission/REPORT.md` (không còn placeholder nào).
- [ ] Đã trả lời trung thực trong `submission/REFLECTION.md`.
- [ ] Chạy `make verify` đạt trạng thái **Ready to submit** (0 FAIL).
- [ ] Đóng gói thư mục nộp bài theo đúng Option lựa chọn (Option A, B, hoặc C).
