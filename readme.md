# bengali-bn-to-en

### How to use

You can use this model directly with a pipeline:


```python
from transformers import AutoTokenizer, pipeline
tokenizer = AutoTokenizer.from_pretrained("shihab17/bengali-bn-to-en")
model = AutoModelForSeq2SeqLM.from_pretrained("shihab17/bengali-bn-to-en")

sentence = 'ম্যাচ শেষে পুরস্কার বিতরণের মঞ্চে তামিমের মুখে মোস্তাফিজের প্রশংসা শোনা গেল'

translator = pipeline("translation_bn_to_en", model=model, tokenizer=tokenizer)
output = translator(sentence)
print(output)
```



This model is a fine-tuned version of [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en) on the kde4 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6885
- Bleu: 50.9475
- Gen Len: 6.7043

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Bleu    | Gen Len |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|:-------:|
| 1.8866        | 1.0   | 2047  | 1.6397          | 39.6617 | 8.0651  |
| 1.5769        | 2.0   | 4094  | 1.6160          | 33.0247 | 8.9865  |
| 1.3622        | 3.0   | 6141  | 1.6189          | 53.483  | 6.6037  |
| 1.2317        | 4.0   | 8188  | 1.6280          | 51.6882 | 6.762   |
| 1.1248        | 5.0   | 10235 | 1.6450          | 53.1619 | 6.5515  |
| 1.0297        | 6.0   | 12282 | 1.6587          | 52.3224 | 6.5905  |
| 0.9632        | 7.0   | 14329 | 1.6733          | 52.3362 | 6.5441  |
| 0.8831        | 8.0   | 16376 | 1.6802          | 49.3544 | 6.8272  |
| 0.8291        | 9.0   | 18423 | 1.6868          | 49.9486 | 6.792   |
| 0.8175        | 10.0  | 20470 | 1.6885          | 50.9475 | 6.7043  |


### Framework versions

- Transformers 4.29.1
- Pytorch 2.0.0+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3