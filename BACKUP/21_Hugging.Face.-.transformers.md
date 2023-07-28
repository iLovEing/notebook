# [Hugging Face - transformers](https://github.com/iLovEing/notebook/issues/21)

# 链接
[参考示例](todo)

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
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
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
res = tokenizer(list_of_sentences, padding="max_length", max_length=20, truncation=True)
  - 返回tensor
inputs = tokenizer(list_of_sentences, padding=True, truncation=True, ***return_tensors="pt"***)

- **加载特殊编码**
某些模型有tokenizer相关的py文件，需要添加***trust_remote_code***参数
tokenizer1 = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", ***trust_remote_code=True***, revision="v1.1.0")

---

# model

- **加载、保存、配置**
model = AutoModel.from_pretrained("hfl/rbt3").to(device)
print(model.config)
*or*
config = AutoConfig.from_pretrained("hfl/rbt3") && print(config)  //可以看到config原型，找到实现有更多的配置项

- **使用**
  - 不带model head
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
inputs = tokenizer(sentencelist, padding=True, return_tensors="pt").to(device)
**output = model(**inputs)
  - 带model head
指定任务可以得到带model head的模型，比如bert可以得到cls token之后的pooler out，各种模型点进去看实现最清楚
clz_model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=5).to(device)