# CHECKPOINT 3 (CP3) — HUẤN LUYỆN LORA CHUẨN (VÙNG KHÔNG HỐI TIẾC)

> **Notebook tương ứng**: `notebooks/03_train_correct.py`  
> **Thời gian thực thi**: ~15–25 phút trên GPU T4 (30 optimizer steps, 2 epochs)  
> **Điểm Rubric liên quan**: 15 điểm (Mục 1.4: 10đ, Mục 2.2: 5đ)

---

## 🎯 1. MỤC TIÊU CỦA CP3
1. **Áp dụng cấu hình "Vùng không hối tiếc" (LoRA Without Regret)**: Tránh các sai lầm phổ biến khi tinh chỉnh LLM; gắn adapter vào toàn bộ các lớp Linear phù hợp, sử dụng đúng thang Learning Rate và kiểm soát batch size.
2. **Huấn luyện Adapter chuẩn (`correct`)**: Tinh chỉnh mô hình trên tập train 225 mẫu CSKH với mặt nạ loss `assistant-only`.
3. **Đo đạc chi phí tài nguyên thực tế**: Theo dõi lượng tham số có thể huấn luyện (Trainable Parameters), thời gian huấn luyện và mức chiếm dụng VRAM đỉnh (Peak VRAM).
4. **Lưu trữ artefact chuẩn tắc**: Xuất adapter weights định dạng `safetensors` và ghi nhật ký vào `results/runs.csv`.

---

## 🛠️ 2. CP3 LÀM NHỮNG VIỆC GÌ?

### Bước 1: Nạp Mô Hình & Tokenizer Theo Tier Phần Cứng
- Nạp base model theo cấu hình môi trường (`T4` $\rightarrow$ `unsloth/Qwen3.5-4B`, float16).
- Đồng bộ tokenizer và cấu hình padding, EOS token.

### Bước 2: Thiết Lập Cấu Hình LoRA Chuẩn Mực
Theo các khuyến nghị trong bài giảng và tài liệu kỹ thuật:
- **`target_modules`**: Đặt là `"text-linear"` (hàm `resolve_target_modules` sẽ quét tất cả các projection: $q, k, v, o, gate, up, down$ của transformer text; không gắn bừa vào vision tower).
- **Rank ($r$) & Alpha ($\alpha$)**: $r = 16$, $\alpha = 16$ hoặc $32$.
- **LoRA Dropout**: $0.05$ (hoặc $0.0$).
- **In `layer_types`**: In ra danh sách các lớp được gắn adapter để xác minh trực quan.

### Bước 3: Cấu Hình Huấn Luyện (SFTConfig)
- **Learning Rate**: $1 \times 10^{-4}$ (thang LR đặc thù của LoRA, lớn hơn 10 lần so với Full Fine-tuning).
- **Effective Batch Size**: Giữ $< 32$ (ví dụ: `per_device_train_batch_size = 4`, `gradient_accumulation_steps = 4` $\rightarrow$ effective batch $= 16$).
- **Số Epochs / Max Steps**: Mặc định 2 epochs ($\approx 30$ optimizer steps).
- **Loss Masking**: Chế độ `assistant-only` (kế thừa từ CP1).

### Bước 4: Chạy Huấn Luyện & Lưu Trữ Kết Quả
- Chạy huấn luyện qua `SFTTrainer`.
- Lưu adapter vào `adapters/correct/`:
  + `adapter_model.safetensors`
  + `adapter_config.json`
- Ghi 1 dòng thông tin vào `results/runs.csv` bao gồm: tên run (`correct`), vị trí gắn (`text-linear`), $r$, số tham số trainable, LR, loss cuối cùng, thời gian huấn luyện (giây), peak VRAM (GB), và `max_steps`.
- Giải phóng VRAM (`generate.free_memory()`) để sẵn sàng cho các lượt tiếp theo.

---

## 🧪 3. TEST VÀ KIỂM THỬ RA SAO?

### 3.1. Lệnh thực thi
```bash
# Chạy notebook huấn luyện:
make nb3

# Hoặc:
python notebooks/03_train_correct.py

# Chạy test kiểm tra khởi tạo model và trainer:
pytest tests/test_modeling_and_train.py -k "test_correct" -v
```

### 3.2. Tiêu chí kiểm tra thành công (Verification Criteria)
1. **Thư mục adapter**:
   - `adapters/correct/adapter_model.safetensors` tồn tại (dung lượng khoảng vài chục MB).
   - `adapters/correct/adapter_config.json` tồn tại và ghi rõ $r=16$, các module đã target.
2. **Tệp nhật ký `results/runs.csv`**:
   - Có dòng với cột `run` là `correct`.
   - Cột `trainable_params` khoảng ~32 triệu tham số (với Qwen3.5-4B).
   - Cột `max_steps` được ghi nhận rõ ràng (ví dụ: 30).
   - `final_loss` hội tụ tốt (thường giảm từ $>2.0$ xuống khoảng $<0.1$).

---

## ⚠️ 4. CẠM BẪY & LỖI THƯỜNG GẶP (GOTCHAS)

> [!CAUTION]
> **Nhầm thang Learning Rate**:
> Không dùng $LR = 1 \times 10^{-5}$ cho LoRA. Thang $10^{-5}$ là dành cho Full FT (khi cập nhật toàn bộ 100% trọng số). Với LoRA chỉ cập nhật một ma trận phụ trợ rank thấp, nếu để $10^{-5}$ thì bước nhảy quá nhỏ, loss sẽ gần như đi ngang và mô hình không học được gì (điều này sẽ được chứng minh ở CP4).

> [!WARNING]
> **Tràn bộ nhớ GPU (OOM) ở run tiếp theo**:
> Sau khi train xong, PyTorch cache có thể vẫn giữ bộ nhớ trên GPU. Cần đảm bảo hàm dọn dẹp bộ nhớ (`torch.cuda.empty_cache()` và `gc.collect()`) được gọi trước khi chuyển sang notebook tiếp theo.

---

## 📝 5. CHECKLIST CHI TIẾT CP3

- [ ] **Môi trường**: Runtime GPU đang hoạt động và nhận diện đúng CUDA device.
- [ ] **Khởi động**: Chạy `make nb3` và theo dõi quá trình loss giảm dần qua từng step.
- [ ] **Kiểm tra Artefact Adapter**:
  - [ ] Thư mục `adapters/correct/` đã được tạo.
  - [ ] Tệp `adapter_model.safetensors` có dung lượng hợp lệ (> 0 byte).
  - [ ] Tệp `adapter_config.json` thể hiện đúng thông số LoRA.
- [ ] **Kiểm tra File Log `results/runs.csv`**:
  - [ ] Xuất hiện dòng `correct`.
  - [ ] Ghi nhận đủ `trainable_params`, `learning_rate`, `final_loss`, `train_time_sec`, `peak_vram_gb`, `max_steps`.
- [ ] **Ghi nhận số liệu**: Copy thông số của run `correct` vào Bảng mục 4 trong `submission/REPORT.md`.
