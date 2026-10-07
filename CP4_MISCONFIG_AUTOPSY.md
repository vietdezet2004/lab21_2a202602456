# CHECKPOINT 4 (CP4) — GIẢI PHẪU CẤU HÌNH SAI (3 RUN ĐỐI CHỨNG CÔNG BẰNG)

> **Notebook tương ứng**: `notebooks/04_misconfig_autopsy.py`  
> **Thời gian thực thi**: ~45–60 phút trên GPU T4 (3 lượt huấn luyện độc lập)  
> **Điểm Rubric liên quan**: 20 điểm (Mục 2.1: 10đ, Mục 2.2: 5đ, Mục 2.3: 5đ)

---

## 🎯 1. MỤC TIÊU CỦA CP4
1. **Thiết kế phép thử đối chứng khoa học chuẩn mực (Fair Controlled Experiment)**: Mỗi run đối chứng chỉ thay đổi **đúng một biến số** duy nhất so với run `correct`.
2. **Khớp ngân sách tham số cho run `attn_only`**: Chứng minh sự vượt trội của vị trí gắn adapter (`text-linear` vs `q,v`) bằng cách nâng rank của `attn_only` lên để số tham số có thể huấn luyện ngang bằng với `correct` ($\text{sai lệch} < 5\%$).
3. **Phân tích tác động của siêu tham số Learning Rate**: Thực nghiệm run `wrong_lr` để quan sát hiện tượng underfitting khi áp dụng thang LR của Full-FT vào LoRA.
4. **Đo đạc sự đánh đổi của QLoRA**: Định lượng chính xác tỷ lệ tiết kiệm VRAM và chi phí về thời gian huấn luyện khi lượng tử hóa 4-bit NF4.
5. **Đồng nhất số bước huấn luyện**: Đảm bảo cả 4 run chạy chính xác cùng một ngân sách số bước (`max_steps`).

---

## 🛠️ 2. CP4 LÀM NHỮNG VIỆC GÌ?

CP4 thực hiện tuần tự 3 lượt huấn luyện đối chứng, ghi log và lưu adapter riêng:

### 2.1. Run 1: `attn_only` (Sai lầm về Vị trí gắn Adapter)
- **Sai lầm phổ biến**: Chỉ gắn LoRA vào các lớp attention $q, v$ thay vì toàn bộ các lớp linear.
- **Cách đối chứng công bằng**: Dùng hàm `matched_rank()` để tính toán rank tương đương.
  - Trên Qwen3.5-4B: $r$ được tự động nâng từ 16 lên $\approx 283$.
  - Tổng số tham số trainable: $\approx 32.45$ triệu (khớp $99.97\%$ với run `correct` $\approx 32.46$ triệu).
- **Mục đích**: Bác bỏ lập luận "rank cao sẽ bù đắp được vị trí hẹp".

### 2.2. Run 2: `wrong_lr` (Sai lầm về Thang đo Tốc độ học)
- Giữ nguyên vị trí `text-linear` và $r=16$.
- Thay đổi duy nhất: Giảm Learning Rate xuống 10 lần: $LR = 1 \times 10^{-5}$ (thang đo thông thường của Full-FT).
- **Mục đích**: Chứng minh đường loss đi ngang hoặc giảm cực kỳ chậm do cập nhật ma trận tích vô hướng rank thấp bị quá yếu.

### 2.3. Run 3: `qlora` (Đánh đổi Bộ nhớ & Tốc độ)
- Nạp base model dưới dạng 4-bit NormalFloat (NF4) thông qua `bitsandbytes`.
- Giữ nguyên vị trí `text-linear`, $r=16$, $LR = 1 \times 10^{-4}$.
- **Mục đích**: Đo đạc mức tiết kiệm VRAM (giảm từ $\approx 12.07\text{ GB}$ xuống $\approx 7.15\text{ GB}$, tiết kiệm $\approx 41\%$) và so sánh tốc độ huấn luyện.

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

### 3.1. Lệnh thực thi
```bash
# Chạy cả 3 run đối chứng:
make nb4

# Hoặc chạy script:
python notebooks/04_misconfig_autopsy.py

# Nếu bị gián đoạn giữa chừng, script sẽ tự động bỏ qua run đã hoàn tất.
# Để chạy lại một run cụ thể:
ONLY=qlora python notebooks/04_misconfig_autopsy.py

# Để ép buộc chạy lại toàn bộ:
FORCE_RETRAIN=1 python notebooks/04_misconfig_autopsy.py
```

### 3.2. Tiêu chí kiểm tra thành công (Verification Criteria)
Mở tệp `results/runs.csv` và kiểm tra các điều kiện nghiêm ngặt:
1. **Đủ 4 dòng**: Phải có đủ 4 dòng đại diện cho `correct`, `attn_only`, `wrong_lr`, `qlora`.
2. **Khớp tham số của `attn_only`**:
   $$\frac{|\text{trainable}(\text{attn\_only}) - \text{trainable}(\text{correct})|}{\text{trainable}(\text{correct})} < 0.05$$
   (Gatekeeper `scripts/verify.py` kiểm tra tự động dòng này).
3. **Đồng nhất số bước huấn luyện**:
   Giá trị cột `max_steps` ở cả 4 dòng phải giống hệt nhau (ví dụ: đều là 30).
4. **VRAM của QLoRA**: Cột `peak_vram_gb` của dòng `qlora` phải giảm rõ rệt so với 3 dòng còn lại (giảm ~40%).

---

## ⚠️ 4. CẠM BẪY & LỖI THƯỜNG GẶP (GOTCHAS)

> [!CAUTION]
> **Bẫy Đánh giá: Xếp hạng các cấu hình bằng Train Loss (Lỗi #3)**:
> Khi nhìn vào `results/runs.csv`, bạn có thể thấy `attn_only` có train loss thấp hơn cả `correct` (ví dụ: $0.0531 < 0.0549$). **TUYỆT ĐỐI KHÔNG DÙNG TRAIN LOSS ĐỂ KẾT LUẬN ATTN_ONLY TỐT HƠN!** Train loss đo khả năng overfit tập train của các tham số tập trung, nhưng ở NB5 khi đo trên tập eval thực tế, `correct` mới là cấu hình cho kết quả vượt trội. Xếp hạng bằng train loss sẽ bị trừ điểm nặng ở rubric 2.5!

> [!WARNING]
> **Lệch ngân sách `max_steps` khi chỉnh `EPOCHS`**:
> Biến môi trường `EPOCHS` (hoặc `max_steps`) phải áp dụng đồng nhất cho cả NB3 và NB4. Không được chạy NB3 với 2 epochs rồi chạy NB4 với 1 epoch vì sẽ vi phạm nguyên tắc so sánh công bằng.

---

## 📝 5. CHECKLIST CHI TIẾT CP4

- [ ] **Khởi động**: Chạy `make nb4` và kiểm tra cả 3 run lần lượt hoàn tất.
- [ ] **Kiểm tra File Log `results/runs.csv`**:
  - [ ] Đủ cả 4 hàng: `correct`, `attn_only`, `wrong_lr`, `qlora`.
  - [ ] Cột `max_steps` của cả 4 hàng có giá trị bằng nhau.
  - [ ] Sai số `trainable_params` giữa `attn_only` và `correct` $< 5\%$.
  - [ ] Ghi nhận `peak_vram_gb` của `qlora` giảm khoảng 40% so với `correct`.
- [ ] **Điền Báo Cáo**:
  - [ ] Cập nhật bảng mục 4 trong `submission/REPORT.md`.
  - [ ] Chuẩn bị trả lời 3 câu hỏi phân tích sâu (4.1, 4.2, 4.3) sau khi có điểm target từ CP5.
