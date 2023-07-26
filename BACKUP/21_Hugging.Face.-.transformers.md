# [Hugging Face - transformers](https://github.com/iLovEing/notebook/issues/21)

# pipline

- **基本使用**
pipe = pipeline("text-classification"， device="cuda")
pipe("i love u")


- **加载本地任务**
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


- **查看支持任务**
pipelines.SUPPORTED_TASKS


- **查看任务参数**
打印pipe，可以看到具体函数，然后打印函数__doc__ or 直接进入。

---

# pipline

- **基本使用**
pipe = pipeline("text-classification"， device="cuda")
pipe("i love u")


- **加载本地任务**
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


- **查看支持任务**
pipelines.SUPPORTED_TASKS


- **查看任务参数**
打印pipe，可以看到具体函数，然后打印函数__doc__ or 直接进入。


---

# tokenizer

- **加载、保存、基本信息**
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", device="cuda")
tokenizer.save_pretrained("./roberta_tokenizer")
tokenizer.special_tokens_map, tokenizer.vocab_size, tokenizer.vocab

- **编码和解码文本**
  - 分步编码
tokens = tokenizer.tokenize(list_of_sentences)
ids = tokenizer.convert_tokens_to_ids(tokens)
tokens = tokenizer.convert_ids_to_tokens(ids)
str_sen = tokenizer.convert_tokens_to_string(tokens)
*or*
ids = tokenizer.encode(sen, padding="max_length", max_length=10, add_special_tokens=True)
str_sen = tokenizer.decode(ids, skip_special_tokens=True)
  - 一步到位（带token_type和attention_mask）
res = tokenizer(sens, padding="max_length", max_length=20, truncation=True)

- **加载特殊编码**
某些模型有tokenizer相关的py文件，需要添加***trust_remote_code***参数
tokenizer1 = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", ***trust_remote_code=True***, revision="v1.1.0")