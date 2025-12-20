import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

MODEL_DIR = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR,output_attentions=True)
model.eval()

# text = "Intel delays new chip production amid manufacturing issues."
# text = "Dell Exits Low-End China Consumer PC Market  HONG KONG (Reuters) - Dell Inc. &lt;DELL.O&gt;, the world's  largest PC maker, said on Monday it has left the low-end  consumer PC market in China and cut its overall growth target  for the country this year due to stiff competition in the  segment."
def print_attention(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions

    # 取最后一层
    last_layer = attentions[-1]
    # 平均所有 heads
    avg_attn = last_layer.mean(dim=1)[0]

    # CLS token 对其他 token 的 attention
    cls_attn = avg_attn[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    token_importance = list(zip(tokens, cls_attn.tolist()))
    print(text)
    for tok, score in sorted(token_importance, key=lambda x: x[1], reverse=True)[:10]:
        print(tok, round(score, 4))


texts = [
    "Intel to delay product aimed for high-definition TVs SAN FRANCISCO -- In the latest of a series of product delays, Intel Corp. has postponed the launch of a video display chip it had previously planned to introduce by year end, putting off a showdown with Texas Instruments Inc. in the fast-growing market for high-definition television displays.",
    "Dell Exits Low-End China Consumer PC Market  HONG KONG (Reuters) - Dell Inc. &lt;DELL.O&gt;, the world's  largest PC maker, said on Monday it has left the low-end  consumer PC market in China and cut its overall growth target  for the country this year due to stiff competition in the  segment.",
    "Some People Not Eligible to Get in on Google IPO Google has billed its IPO as a way for everyday people to get in on the process, denying Wall Street the usual stranglehold it's had on IPOs. Public bidding, a minimum of just five shares, an open process with 28 underwriters - all this pointed to a new level of public participation. But this isn't the case.",
    "Venezuela Prepares for Chavez Recall Vote Supporters and rivals warn of possible fraud; government says Chavez's defeat could produce turmoil in world oil market.",
    "Promoting a Shared Vision As Michael Kaleko kept running into people who were getting older and having more vision problems, he realized he could do something about it."
]

for t in texts:
    print_attention(t)