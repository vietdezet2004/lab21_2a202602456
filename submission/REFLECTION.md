# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là run `attn_only` (chỉ gắn vào q, v) khi được nâng rank lên 283 để bằng số tham số với `correct` thì có train loss thấp hơn (0.5363 so với 0.6265), nhưng điểm target thực tế lại không hề vượt trội hơn. Nó chứng minh trực quan rằng việc tối ưu loss cục bộ trên tập train dễ gây ảo tưởng về năng lực mô hình.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu sinh suy luận đánh giá trên tập eval (NB2 và NB5) chứ không phải ở khâu huấn luyện LoRA (NB3). Ban đầu tôi dự đoán việc train model sẽ tốn thời gian nhất, nhưng thực tế việc greedy decode 3 lần cho cả 50 mẫu tốn tới hơn 35 phút trên T4 GPU.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước đây tôi tin rằng hễ fine-tune đạt accuracy cao (như 97% ở bài toán này) là mô hình đã hoàn hảo và sẵn sàng mang đi phục vụ người dùng. Giờ tôi hiểu rằng fine-tune có thể âm thầm hủy hoại năng lực tổng quát (quên thảm họa -13.6% điểm regression) nếu không có cổng kiểm soát hồi quy chặt chẽ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để giải thích các thông số trong pipeline, hướng dẫn cấu hình môi trường Colab và phân tích nguyên nhân tại sao run `wrong_lr` bị sụp đổ. AI từng gợi ý tôi chỉ cần nhìn vào train loss để đánh giá mô hình, và tôi đã phát hiện ra điều đó sai hoàn toàn khi đối chiếu với rubric của bài lab.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở code ra train ngay, mà là đóng băng một tập đánh giá chuẩn và xây dựng một prompt engineering thật tối ưu (Baseline b) để đo đạc mốc năng lực ban đầu. Nếu prompt tối ưu đã giải quyết tốt 80–90% bài toán với chi phí rẻ hơn, tôi sẽ cân nhắc kỹ trước khi quyết định tốn tài nguyên fine-tune.
