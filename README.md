# text-classification
apply transformer based nlp model to text classification

## Dateset
- ag_news

## Methodology
- Text preprocessing and data splitting
- Fine-tuning a pre-trained BERT model
- Evaluation using accuracy and F1-score

## Result
### Different max_length
| max_length   | Accuracy     | Macro F1     | Eval Time (s)   |
|--------------|--------------|--------------|-----------------|
| 64           | 0.9461       | 0.9461       | 5.5             |
| 128          | 0.9479       | 0.9480       | 10.4            |
| 256          | 0.9468       | 0.9469       | 20.4            |
| 512          | 0.9479       | 0.9479       | 41.4            |

* max_length=64: Shorter input length leads to information loss, as important contextual cues may be truncated.
* max_length=128: This suggests that most informative content is captured within the first 128 tokens, and additional context provides diminishing returns.
* max_length=256: Increasing context length introduces more padding and irrelevant information, which may dilute useful signals and make optimization harder.
* max_length=512: While longer context theoretically preserves more information, empirical results show no consistent performance gains, indicating that long-range dependencies are not critical for this task.

### Effect of Input Length
Results show that increasing max_length from 64 to 128 significantly improves performance, suggesting that short contexts fail to capture sufficient semantic information. 
However, further increasing the input length to 256 or 512 does not lead to consistent gains, while substantially increasing computational cost. 
This indicates that most task-relevant information is contained within the first 128 tokens, and excessive context may introduce noise and reduce training efficiency.

### Different epoch(max_length=128)
| Epoch        | Eval Loss    | Accuracy     | Macro F1        |
|--------------|--------------|--------------|-----------------|
| 2            | 0.1907       | 0.9482       | 0.9482          |
| 3            | 0.1895       | 0.9479       | 0.9480          |
| 4            | 0.1881       | 0.9475       | 0.9475          |

### Effect of Training Epochs
Results show that performance peaks at epoch 2, with no further improvement observed in later epochs.
Although evaluation loss continues to decrease, accuracy and macro-F1 slightly decline, indicating early overfitting.
Therefore, we select epoch 2 as the optimal training point.

## Error Analysis
Attentions of [CLS]->tokens of misclassified texts:
1. Intel to delay product aimed for high-definition TVs SAN FRANCISCO -- In the latest of a series of product delays, Intel Corp. has postponed the launch of a video display chip it had previously planned to introduce by year end, putting off a showdown with Texas Instruments Inc. in the fast-growing market for high-definition television displays.
- 0.1199
- 0.1175
francisco 0.0519
to 0.0386
##s 0.0374
intel 0.035
. 0.0328
, 0.0325
, 0.032
[SEP] 0.0303

2. Dell Exits Low-End China Consumer PC Market  HONG KONG (Reuters) - Dell Inc. &lt;DELL.O&gt;, the world's  largest PC maker, said on Monday it has left the low-end  consumer PC market in China and cut its overall growth target  for the country this year due to stiff competition in the  segment.
- 0.0562
; 0.0533
exits 0.0514
[CLS] 0.0484
& 0.0426
; 0.0359
said 0.0274
it 0.027
dell 0.0262
market 0.025

3. Some People Not Eligible to Get in on Google IPO Google has billed its IPO as a way for everyday people to get in on the process, denying Wall Street the usual stranglehold it's had on IPOs. Public bidding, a minimum of just five shares, an open process with 28 underwriters - all this pointed to a new level of public participation. But this isn't the case.
google 0.1689
google 0.0774
[CLS] 0.041
. 0.0388
[SEP] 0.0369
. 0.0365
but 0.0362
, 0.0361
' 0.0321
has 0.0274

4. Venezuela Prepares for Chavez Recall Vote Supporters and rivals warn of possible fraud; government says Chavez's defeat could produce turmoil in world oil market.
[SEP] 0.4653
recall 0.1793
vote 0.0664
oil 0.0437
market 0.0386
chavez 0.0332
[CLS] 0.0307
venezuela 0.0255
chavez 0.0212
; 0.017

5. Promoting a Shared Vision As Michael Kaleko kept running into people who were getting older and having more vision problems, he realized he could do something about it.
vision 0.2119
promoting 0.0926
vision 0.0683
michael 0.068
[SEP] 0.0653
shared 0.0598
, 0.0597
a 0.0541
[CLS] 0.051
he 0.0455

### Common Error Patterns
#### 1. Diffuse Attention Distribution
In contrast to correctly classified samples, misclassified instances tend to exhibit diffuse attention patterns.
A significant portion of attention is assigned to:
* Structural tokens (e.g., [SEP], [CLS])
* Punctuation symbols (e.g., -, ;, ,)
* Function words (e.g., to, it, said)
This suggests that the model struggles to form a focused semantic representation when summarizing these inputs.

#### 2. Multi-Topic Interference
Several misclassified samples involve multiple overlapping themes, such as:
* Technology events combined with business strategy
* Political developments with economic consequences
* Corporate actions framed within broader social narratives
Attention analysis reveals that the model distributes focus across competing themes rather than prioritizing a single dominant semantic structure.
This prevents the [CLS] representation from converging to a clear category-specific summary.

### Interpretation
These findings indicate that classification errors are not primarily caused by insufficient context length or lack of lexical knowledge.
Instead, they stem from the model’s limited ability to:
* Prioritize discriminative event-level semantics
* Resolve semantic ambiguity in multi-topic narratives
* Suppress structurally salient but semantically uninformative tokens

### Summary
Misclassification primarily arises from diffuse attention and semantic ambiguity in multi-topic news articles, rather than from insufficient context length or model capacity.