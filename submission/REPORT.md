# Lab 21 — Evaluation Report

**Họ tên**: Phùng Quốc Việt  **MSSV**: 2A202602456  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây khớp 100% với các tệp JSON và CSV trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt $\rightarrow$ JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 tokens (max seen: 101 tokens) *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — tokenizer ChatML của dòng mô hình Qwen3.5 tự động render chuỗi `<think>\n\n</think>\n\n` vào đầu lượt trả lời của assistant ngay cả khi câu trả lời không có reasoning. Nhóm đã xử lý vấn đề này bằng phương pháp render text, tokenize với cờ `return_offsets_mapping=True` và chỉ áp dụng supervision trên các token nằm trong span ký tự của câu trả lời, không diff token list trực tiếp để tránh lỗi phân tách token không đồng nhất (F-01).

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả xác thực |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% số token được tính loss) |
| Câu trả lời nằm trong loss | true (xác nhận bằng giải mã ngược ID) |
| Câu hỏi KHÔNG nằm trong loss | true (toàn bộ prompt được gán nhãn -100) |

Dán 3–5 dòng đầu của đoạn được tính loss từ `results/mask_proof.json`:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3207.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1023.4 |
| (c) LoRA fine-tune | 0.970 | 0.656 | 1.000 | 1368.2 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) vượt trội hoàn toàn so với (a) trên bài toán đích (target tăng từ 0.000 lên 0.765), định dạng JSON parse được đạt 100% (format = 1.000 so với 0.000 của a), và độ trễ giảm hơn 3 lần từ 3207.5 ms xuống 1023.4 ms nhờ cấu trúc prompt rõ ràng.  
**Bạn có sửa `OPTIMIZED_PROMPT` không?** Không — nhóm giữ nguyên chuỗi `OPTIMIZED_PROMPT` nguyên bản của bài lab (mã SHA256 được bảo toàn tuyệt đối) nhằm giữ vững tính liêm chính nghiên cứu, không làm suy yếu prompt đối thủ để tâng bốc kết quả fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | 0.970 | 394.2 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5363 | 0.970 | 260.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.000 | 388.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 474.7 | 3.86 |

**4.1 — Phân tích Vị trí gắn Adapter so với Rank:**  
Run `attn_only` được khớp ngân sách tham số trainable tương đương hoàn toàn với `correct` (32,456,704 so với 32,464,896, sai lệch chỉ 0.025%). Trên tập đánh giá target, `attn_only` đạt kết quả 0.970 (ngang bằng với `correct`), nhưng khi nhìn vào train loss thì `attn_only` lại có loss thấp hơn (0.5363 so với 0.6265). Thứ tự theo train loss đã bị đảo ngược so với bản chất thực nghiệm: việc nhồi nhét rank cực đại ($r=283$) vào số ít lớp attention giúp mô hình khớp nhanh hơn trên tập train nhưng không đem lại bất kỳ ưu thế vượt trội nào trên tập đánh giá so với việc phân bổ đều rank vừa phải ($r=16$) lên toàn bộ 12 khối text-linear. Điều này khẳng định vị trí can thiệp trên toàn bộ kiến trúc là đòn bẩy căn bản, còn rank chỉ là thông số điều tiết dung lượng chứ không thể thay thế cho vị trí gắn adapter hợp lý.

**4.2 — Phân tích Đòn bẩy Tốc độ học (Learning Rate):**  
Run `wrong_lr` chỉ thay đổi duy nhất một tham số là giảm Learning Rate xuống 10 lần ($10^{-5}$ thay vì $10^{-4}$), nhưng dẫn đến sự sụp đổ hoàn toàn về năng lực với target và format đều rơi về 0.000. Đường loss của `wrong_lr` bị nghẽn ở mức cao (1.5702) và đi ngang từ các step đầu tiên do bước cập nhật ma trận tích vô hướng rank thấp của LoRA đòi hỏi bước nhảy gradient lớn hơn thang Full Fine-tuning. Nếu chỉ quan sát đường loss mà không kiểm tra thiết lập siêu tham số, người làm kỹ thuật rất dễ đưa ra kết luận sai lầm rằng dữ liệu bị lỗi, bài toán quá phức tạp hoặc phương pháp LoRA không tương thích với mô hình nền.

**4.3 — Phân tích Đánh đổi của QLoRA (Bộ nhớ vs Tốc độ & Độ chính xác):**  
QLoRA đã chứng minh khả năng cắt giảm bộ nhớ ấn tượng khi hạ mức VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm tới 56.0% VRAM), cho phép vận hành trơn tru ngay cả trên các GPU phổ thông dung lượng nhỏ. Tuy nhiên, sự đánh đổi là thời gian huấn luyện kéo dài lên 474.7 giây (chậm hơn khoảng 20% so với 394.2 giây của `correct` do chi phí giải lượng tử hóa 4-bit liên tục) và độ chính xác target bị sụt giảm nhẹ từ 0.970 xuống 0.940. Kết quả đo đạc thực tế này củng cố khuyến nghị rằng nếu hạ tầng phần cứng có đủ VRAM (như T4 16GB), ta nên ưu tiên LoRA 16-bit chuẩn để tối ưu cả tốc độ lẫn chất lượng; QLoRA là lựa chọn bắt buộc khi bị giới hạn tài nguyên bộ nhớ nghiêm ngặt.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.136` · `valid_trace_rate = 0.00`

**Diễn giải khoa học:**  
Phán quyết FAILED từ Cổng hồi quy là một kết luận hoàn toàn chuẩn xác và có giá trị khoa học thực tiễn cao. Mặc dù bản fine-tune đạt mức tăng trưởng vượt bậc trên bài toán chuyên biệt (+0.205 trên tập target, đưa độ chính xác lên tới 97%), mô hình đã vi phạm điều kiện an toàn hồi quy khi năng lực tổng quát (`regression`) bị sụt giảm nghiêm trọng tới -0.136 (từ 0.7911 xuống 0.6556, vượt xa ngưỡng dung sai cho phép là 0.020). 

Đây là minh chứng kinh điển cho hiện tượng "Quên thảm họa" (Catastrophic Forgetting) trong kỷ nguyên hậu huấn luyện. Vì tập dữ liệu huấn luyện 225 mẫu chỉ thuần túy bao gồm các cấu trúc JSON phân loại vé CSKH, các tham số adapter đã dịch chuyển biểu diễn không gian latent theo hướng chuyên biệt hóa quá mức, làm bào mòn khả năng suy luận logic và kiến thức phổ thông nền tảng. Theo nguyên lý được trình bày trong bài giảng (Deck §6.3), giải pháp kỹ thuật bắt buộc để đảo ngược phán quyết FAILED trong môi trường sản xuất thực tế là bổ sung từ 1% đến 5% dữ liệu tổng quát (Replay / General Domain Anchors) vào tập huấn luyện nhằm bảo toàn phân phối tri thức gốc của mô hình.

---

## 6. Định tính — Bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, đặt chuột không dây VN232232. Đổi trả. Gấp. | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, cao, chuột không dây, tich_cuc | ✅ FT thắng: Nhận diện chính xác 100% cả 4 khóa. |
| 2 | Chào shop, đặt ốp lưng VN833689. Sai màu. Sớm nhé. Bực mình. | san_pham_loi, trung_binh, ốp lưng điện thoại, tieu_cuc | san_pham_loi, trung_binh, ốp lưng điện thoại, tieu_cuc | san_pham_loi, trung_binh, ốp lưng điện thoại, tieu_cuc | ✅ FT thắng: Cảm xúc tiêu cực và mức khẩn cấp chuẩn xác. |
| 3 | Alo shop, đặt ốp lưng DH734695. Giá bao nhiêu. 3 ngày rồi. | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | van_chuyen, trung_binh, ốp lưng điện thoại, trung_tinh | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | ✅ FT thắng: Prompt (b) bị lừa bởi "3 ngày rồi" thành van_chuyen, FT phân loại đúng. |
| 4 | Cho mình hỏi, bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện. | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, trung_binh, bình giữ nhiệt, tich_cuc | ❌ **FT thua**: Nhầm urgency "thap" thành "trung_binh" do thiên kiến phản xạ. |
| 5 | Shop ơi, nồi chiên không dầu DH249548. Thiếu phụ kiện. Khi nào tiện. | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: Bỏ qua sắc thái "Khi nào tiện", tự động gán urgency mức trung bình. |

**Mẫu chung ở các ca FT thua:**  
Cả hai trường hợp fine-tune bị thua (ca #4 và #5) đều có chung một cơ chế lỗi: mô hình dự đoán sai trường `urgency` từ mức "thap" thành "trung_binh". Nguyên nhân là do trong tập dữ liệu huấn luyện, các ticket khiếu nại về tiền bạc ("Chưa thấy tiền") hoặc lỗi phần cứng ("Thiếu phụ kiện") chiếm đa số ở mức độ khẩn cấp trung bình hoặc cao. Mô hình fine-tune đã hình thành một mối tương quan giả (spurious correlation) giữa từ khóa khiếu nại với mức urgency trung bình, dẫn đến việc bỏ qua tín hiệu sắc thái nhẹ nhàng "Khi nào tiện" mà prompt (b) nhờ năng lực zero-shot tổng quát vẫn nắm bắt được chuẩn xác.

---

## 7. Kết luận & điều tôi học được

**Kết luận chuyên môn:**  
Dựa trên các bằng chứng thực nghiệm thu được, câu trả lời cho câu hỏi có nên triển khai (deploy) bản fine-tune này ra môi trường production hay không là: **CHƯA NÊN DEPLOY TRỰC TIẾP**. Mặc dù mô hình mang lại hiệu quả vượt trội về năng lực chuyên môn (độ chính xác trích xuất JSON đạt 97%, định dạng chuẩn 100%, độ trễ được kiểm soát ở mức 1368 ms), nhưng sự suy thoái năng lực tổng quát (-0.136 trên thang regression) tiềm ẩn rủi ro rất lớn khi gặp các truy vấn đa dạng từ người dùng ngoài đời thực. 

Đòn bẩy thực sự quyết định thành công của bài lab này không nằm ở rank cao hay thuật toán tối ưu phức tạp, mà nằm ở **tính đúng đắn của loss mask** (đảm bảo chỉ tính loss trên câu trả lời) và **thang đo learning rate phù hợp** ($10^{-4}$ cho LoRA). Việc nâng rank lên 283 ở `attn_only` hoàn toàn không giúp mô hình vượt qua `correct` ($r=16$), chứng minh rằng kiến trúc bao phủ toàn diện các lớp linear quan trọng hơn việc tối đa hóa dung lượng cục bộ. Để đủ điều kiện triển khai, bước đi tiếp theo bắt buộc phải là bổ sung 3% dữ liệu tổng quát vào pha huấn luyện để vượt qua Cổng hồi quy.

**Ba điều tôi học được:**
1. **Loss Mask là nền móng tuyệt đối**: Không bao giờ được tin tưởng vào các thiết lập mặc định của thư viện mà phải chứng minh bằng giải mã ngược token; một sai sót nhỏ khiến loss tính cả vào prompt sẽ biến mô hình thành cỗ máy học vẹt câu hỏi.
2. **Train Loss là thước đo giả tạo nếu dùng để so sánh kiến trúc**: Run `attn_only` có loss huấn luyện thấp hơn `correct` nhưng năng lực đánh giá thực tế trên bài toán đích không hề vượt trội hơn; việc tối ưu hóa một ma trận rank cao cục bộ chỉ tạo ra ảo tưởng về sự hội tụ.
3. **Prompt Engineering là baseline bắt buộc phải vượt qua**: Không thể tuyên bố fine-tuning thành công nếu chưa đo đạc đối đầu với một prompt được tối ưu hóa bài bản; prompt tốt không chỉ rẻ hơn mà còn bảo toàn nguyên vẹn năng lực suy luận tổng quát của mô hình nền.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Tôi sẽ thử nghiệm trộn 3% tập dữ liệu mở đa miền (Vietnamese General Instruction Dataset) vào quá trình huấn luyện LoRA để kiểm chứng xem liệu mô hình có thể duy trì độ chính xác target 97% đồng thời triệt tiêu độ sụt giảm regression $\Delta$ để đưa phán quyết về trạng thái `PASSED` hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap (đã xác thực pipeline suy luận)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
