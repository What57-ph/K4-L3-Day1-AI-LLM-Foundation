# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**
**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> _Câu trả lời của bạn_
> Ở temperature 0.0, phản hồi thường ổn định, súc tích và gần giống nhau giữa các lần chạy; khi tăng lên 0.5 và 1.0, cách diễn đạt trở nên đa dạng và sáng tạo hơn, đôi khi có thêm các chi tiết phụ. Ở temperature 1.5, mức ngẫu nhiên cao hơn khiến phản hồi dễ xuất hiện câu chữ kém mạch lạc, lặp lại hoặc thông tin ít liên quan hơn. Nhìn chung, temperature càng cao thì độ đa dạng tăng nhưng tính ổn định và nhất quán có xu hướng giảm.


### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> _Câu trả lời của bạn_
> Với chatbot hỗ trợ khách hàng, em sẽ đặt temperature thấp, khoảng 0.3 vì độ chính xác quan trọng hơn sáng tạo. Khách hỏi về chính sách đổi trả, giá cước, hướng dẫn sử dụng... thì câu trả lời cần nhất quán và đúng, không cần sáng tạo hay diễn đạt mới lạ mỗi lần. Nếu hai khách hàng hỏi cùng một câu, temperature thấp giúp họ nhận được câu trả lời tương tự nhau — tránh tình huống một người được thông tin A, người khác lại nhận thông tin có sắc thái khác dù cùng một chính sách. Đồng thời làm giảm rủi ro hallucination, temperature cao làm tăng khả năng model bịa thông tin (sai chính sách, sai số liệu). 0.0 tuyệt đối đôi khi hơi máy móc. Một chút temperature (0.3) vẫn giữ được sự linh hoạt nhẹ trong cách diễn đạt (để câu trả lời không lặp lại y hệt như robot), mà không đánh đổi độ chính xác.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> _Câu trả lời của bạn_
> Ước tính chi phí (chỉ tính phần output, vì đề bài chỉ cho token đầu ra):

Tổng token/ngày = 10.000 người × 3 lần × 350 token = 10.500.000 token = 10.500 (nghìn token)
GPT-4o: 10.500 × 0,010 = $105/ngày
GPT-4o-mini: 10.500 × 0,0006 = $6,3/ngày
Tỷ lệ: 105 / 6,3 ≈ 16,7 lần
GPT-4o xứng đáng với chi phí khi tác vụ đòi hỏi suy luận phức tạp, nhiều bước, hoặc độ chính xác cao ảnh hưởng trực tiếp đến quyết định — ví dụ phân tích hợp đồng pháp lý, debug logic nghiệp vụ phức tạp, hoặc trả lời câu hỏi kỹ thuật chuyên sâu mà sai sót gây hậu quả lớn. Ở đây, chất lượng cao hơn của gpt-4o giúp giảm số lần khách phải hỏi lại và đảm bảo sai sót ở mức thấp — tiết kiệm chi phí vận hành nhiều hơn phần chênh lệch giá API.

## Nên dùng mini đối với các tác vụ lặp đi lặp lại, câu hỏi đơn giản, có cấu trúc rõ ràng — ví dụ trả lời FAQ, tra cứu trạng thái đơn hàng, phân loại ý định trước khi route sang luồng xử lý khác. Với workload như đề bài, phần lớn traffic thực tế thường rơi vào nhóm này — nên một chiến lược phổ biến là dùng mini làm mặc định, chỉ escalate lên gpt-4o khi mini không đủ (ví dụ dựa trên độ dài câu hỏi, từ khóa, hoặc điểm confidence thấp).

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> _Câu trả lời của bạn_
> Với persona "giáo viên tiểu học", phản hồi thường ngắn hơn, dùng từ vựng đơn giản, tránh thuật ngữ kỹ thuật (thay "mã hóa", "phi tập trung" bằng ví dụ đời thường như "cuốn sổ mà ai cũng có một bản giống hệt nhau"), và hay dùng ví dụ cụ thể, gần gũi để minh họa. Với persona "chuyên gia tài chính", phản hồi thường dài và dày đặc thuật ngữ hơn (consensus, hash, sổ cái phân tán, smart contract), đi thẳng vào ứng dụng tài chính (giao dịch, DeFi, tính minh bạch).

> Điều này cho thấy system prompt không thay đổi sự thật mà model biết, mà thay đổi cách trình bày: nó định hình giọng văn, mức độ trừu tượng, và loại ví dụ được chọn vì system message được model coi là vai trò chỉ dẫn xuyên suốt, chi phối cách nó lọc và diễn đạt lại cùng một kiến thức nền cho từng đối tượng khác nhau.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> _Câu trả lời của bạn_
> Token thật (tiktoken): 116
> Ước lượng theo Part 1 (số từ / 0,75): 99 / 0,75 ≈ 132
> Ước lượng theo ký tự/4 (Task 2.1 fallback): 190
> Chênh lệch với ước lượng ký tự/4: (132 - 116) / 116 × 100≈ 13.8%
> Vì sao tiếng Việt tốn nhiều token hơn: Tiếng Việt có thể cần nhiều token hơn tiếng Anh có độ dài tương đương vì tokenizer chia văn bản thành các token/subword dựa trên những mẫu ký tự đã học; các từ và âm tiết tiếng Việt có dấu hoặc ít xuất hiện trong vocabulary có thể bị tách thành nhiều token hơn. Vì vậy, đếm số từ rồi quy đổi bằng một tỷ lệ cố định chỉ là ước lượng và có thể khác đáng kể so với tokenizer thực tế.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> _Câu trả lời của bạn_
> Streaming quan trọng nhất trong các ứng dụng cần **phản hồi theo thời gian thực** như chatbot, trợ lý AI hoặc hệ thống sinh nội dung dài, vì người dùng có thể thấy kết quả từng phần ngay khi model tạo ra thay vì phải chờ toàn bộ phản hồi hoàn tất, từ đó cải thiện trải nghiệm và giảm cảm giác độ trễ. Ngược lại, **non-streaming** phù hợp hơn khi ứng dụng cần nhận **toàn bộ kết quả trước khi xử lý**, chẳng hạn phân tích dữ liệu, tạo JSON có cấu trúc, lưu phản hồi vào cơ sở dữ liệu, kiểm tra nội dung hoặc thực hiện các tác vụ tự động phía backend, vì cách này đơn giản hơn trong việc xử lý và kiểm tra kết quả hoàn chỉnh.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> _Câu trả lời của bạn_

> Delay cố định khiến hàng nghìn client cùng gặp lỗi sẽ đồng loạt retry ở đúng cùng một thời điểm (ví dụ sau 1 giây), tạo ra một đợt sóng request y hệt đợt đầu dội thẳng vào server ngay khi nó vừa có cơ hội hồi phục, khiến sự cố lặp lại theo chu kỳ. So với delay cố định, exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, giúp giảm tần suất retry và cho API thêm thời gian phục hồi khi đang quá tải. Nếu hàng nghìn client cùng sử dụng delay cố định, chúng có thể retry đồng loạt sau cùng một khoảng thời gian và tạo ra một đợt tải lớn mới, khiến server tiếp tục quá tải. Tuy nhiên, exponential backoff một mình vẫn có thể khiến các client retry đồng bộ, vì vậy trong hệ thống thực tế thường kết hợp thêm **jitter** — một khoảng trễ ngẫu nhiên — để phân tán request theo thời gian.

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> _Câu trả lời của bạn_
> System prompt: "Bạn là trợ lý chăm sóc khách hàng của một cửa hàng bán đồ gia dụng trực tuyến.
> Trả lời bằng tiếng Việt, ngắn gọn, tối đa 3-4 câu mỗi lượt.
> Nếu không chắc chắn về thông tin (giá, tồn kho, chính sách đổi trả), hãy nói rõ
> là cần xác nhận thêm thay vì đoán.
> Giữ giọng điệu thân thiện, lịch sự, xưng "mình" và gọi khách là "bạn"."
> Giải thích:
> "Trả lời bằng tiếng Việt" — chỉ định rõ ràng thay vì để model tự suy ra từ ngôn ngữ của câu hỏi. Nếu khách gõ nhầm tiếng Anh xen tiếng Việt, hoặc câu hỏi ngắn/mơ hồ, model có thể "đoán sai" ngôn ngữ phản hồi nếu không có chỉ định tường minh."
> "Ngắn gọn, tối đa 3-4 câu" — không chỉ để tiết kiệm chi phí (ít token output hơn = rẻ hơn), mà còn vì hành vi thực tế của người dùng CSKH: họ đọc lướt, muốn câu trả lời nhanh, không muốn đọc một đoạn văn dài như bài luận cho một câu hỏi đơn giản như "cái này còn hàng không ạ?".
> "Nếu không chắc chắn... hãy nói rõ là cần xác nhận thêm thay vì đoán" — đây là dòng quan trọng nhất để chống hallucination. Không có chỉ dẫn này, model có xu hướng tự tin bịa ra một câu trả lời nghe hợp lý (ví dụ đoán đại một mức giá) thay vì thừa nhận không biết — trong CSKH thực tế, một câu trả lời sai về giá/chính sách gây hại nhiều hơn một câu "để mình kiểm tra lại".
> "Xưng 'mình', gọi khách là 'bạn'" — cụ thể hóa xưng hô thay vì để mặc định, vì tiếng Việt có hệ thống xưng hô phức tạp (anh/chị/bạn/mình/em...) và nếu không chỉ định, model có thể đổi cách xưng hô giữa các câu trong cùng một hội thoại, tạo cảm giác thiếu nhất quán/chuyên nghiệp.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> _Câu trả lời của bạn_

> Hạn chế lớn nhất của trợ lý hiện tại là history bị cắt cứng còn 3 lượt mà không phân biệt thông tin quan trọng hay không, nên nếu khách nêu một chi tiết cần nhớ (như mã đơn hàng) ở đầu cuộc trò chuyện, nó sẽ biến mất khỏi ngữ cảnh sau vài lượt và khách phải lặp lại. Cải thiện đề xuất là tách riêng một "bộ nhớ ngắn hạn" (`session_facts`) không bị xén: sau mỗi lượt, dùng `call_openai_mini` để trích xuất nhanh các thông tin đáng nhớ từ tin nhắn khách thành JSON, gộp vào `session_facts`, rồi chèn thông tin này vào một system message riêng (đặt ngoài phần history bị cắt, giống cách persona luôn được giữ lại) khi gọi model chính. Cách này tốn thêm một lời gọi API rẻ mỗi lượt nhưng đảm bảo các chi tiết định danh quan trọng không bị mất dù hội thoại kéo dài.

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
