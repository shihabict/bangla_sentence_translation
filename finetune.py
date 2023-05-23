import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from datasets import DatasetDict, Dataset, load_metric


class BengaliTranslation:
    def __init__(self, pretrain_checkpoint, data_path, push_to_hub):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_checkpoint)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_checkpoint)
        # self.pretrain_model =
        self.data_path = data_path
        self.max_input_length = 512
        self.max_target_length = 512
        self.source_lang = "bn"
        self.target_lang = "en"
        self.batch_size = 16
        self.num_epochs = 20
        self.push_to_hub = push_to_hub
        self.metric = load_metric("sacrebleu")

    def load_custom_data(self):
        custom_data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            files = file.readlines()
        print(f"Data Size :{len(files)}")
        for line in files:
            if len(line.split('###')[0].strip()) > 1:
                custom_data.append(str({'bn': line.split('###')[0].strip(), 'en': line.split('###')[1].strip()}))
        train_df = pd.DataFrame(custom_data, columns=['translation'])
        train_data = Dataset.from_dict(train_df)
        return DatasetDict({"train": train_data})
        # return data

    def preprocess_function(self, examples):
        prefix = ''
        # examples = eval(examples)
        inputs = [prefix + eval(ex)[self.source_lang] for ex in examples["translation"]]
        targets = [eval(ex)[self.target_lang] for ex in examples["translation"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def finetune(self):
        split_dataset = self.load_custom_data()
        split_datasets = split_dataset["train"].train_test_split(train_size=0.9, seed=20)
        split_datasets["validation"] = split_datasets.pop("test")
        split_datasets['train'] = split_datasets['train'].map(self.preprocess_function, batched=True)
        split_datasets['validation'] = split_datasets['validation'].map(self.preprocess_function, batched=True)
        print(0)
        from transformers.keras_callbacks import PushToHubCallback

        args = Seq2SeqTrainingArguments(
            f"bengali-{self.source_lang}-to-{self.target_lang}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.num_epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=self.push_to_hub,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.pretrain_model)

        trainer = Seq2SeqTrainer(
            self.pretrain_model,
            args,
            train_dataset=split_datasets["train"],
            eval_dataset=split_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.save_model('bn-to-en')

        # trainer.push_to_hub()


if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-bn-en"
    # data_path = 'DATA/new_dataset.txt'
    data_path = 'DATA/combined_data.txt'
    push_to_hub = True
    bengali_translator = BengaliTranslation(model_checkpoint, data_path, push_to_hub)
    bengali_translator.finetune()
