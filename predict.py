from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM


def predict_with_local_model():
    tokenizer = AutoTokenizer.from_pretrained("bengali-bn-to-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("bengali-bn-to-en")

    sentence = 'ম্যাচ শেষে পুরস্কার বিতরণের মঞ্চে তামিমের মুখে মোস্তাফিজের প্রশংসা শোনা গেল'

    translator = pipeline("translation_bn_to_en", model=model, tokenizer=tokenizer)
    output = translator(sentence)
    print(output)
    # print(results[0]['translation_text'])


def predict_with_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("shihab17/bengali-bn-to-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("shihab17/bengali-bn-to-en")

    sentence = 'ম্যাচ শেষে পুরস্কার বিতরণের মঞ্চে তামিমের মুখে মোস্তাফিজের প্রশংসা শোনা গেল'

    translator = pipeline("translation_bn_to_en", model=model, tokenizer=tokenizer)
    output = translator(sentence)
    print(output)
    # print(results[0]['translation_text'])


if __name__ == '__main__':
    predict_with_local_model()
