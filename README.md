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
| ------------ | ------------ | ------------ | --------------- |

* max_length=64: Shorter input length leads to information loss, as important contextual cues may be truncated.
* max_length=128: This suggests that most informative content is captured within the first 128 tokens, and additional context provides diminishing returns.
* max_length=256: Increasing context length introduces more padding and irrelevant information, which may dilute useful signals and make optimization harder.
* max_length=512: While longer context theoretically preserves more information, empirical results show no consistent performance gains, indicating that long-range dependencies are not critical for this task.

### Effect of Input Length
Results show that increasing max_length from 64 to 128 significantly improves performance, suggesting that short contexts fail to capture sufficient semantic information. 
However, further increasing the input length to 256 or 512 does not lead to consistent gains, while substantially increasing computational cost. 
This indicates that most task-relevant information is contained within the first 128 tokens, and excessive context may introduce noise and reduce training efficiency.

### Different epoch
| Epoch        | Eval Loss    | Accuracy     | Macro F1        |
|--------------|--------------|--------------|-----------------|
| 2            | 0.1907       | 0.9482       | 0.9482          |
| 3            | 0.1895       | 0.9479       | 0.9480          |
| 4            | 0.1881       | 0.9475       | 0.9475          |
| ------------ | ------------ | ------------ | --------------- |

### Effect of Training Epochs
Results show that performance peaks at epoch 2, with no further improvement observed in later epochs.
Although evaluation loss continues to decrease, accuracy and macro-F1 slightly decline, indicating early overfitting.
Therefore, we select epoch 2 as the optimal training point.
