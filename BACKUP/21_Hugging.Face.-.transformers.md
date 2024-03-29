# [Hugging Face - transformers](https://github.com/iLovEing/notebook/issues/21)

# 链接
[原视频课程](https://www.bilibili.com/video/BV1ma4y1g791/?spm_id_from=333.788&vd_source=f6617adfe045a4caffc5f65ac3691c24)
[示例代码](https://github.com/iLovEing/sample_code/tree/main/transformers)

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

---

# dataset

- **加载数据集**
dataset = load_dataset("madao33/new-title-chinese")
dataset = load_dataset("super_glue", "boolq") //某些dataset是一个任务集合，需要添加子项
dataset = load_dataset("madao33/new-title-chinese", split="train") //加载某一个split
dataset["train"].column_names, dataset["train"].features //属性

- **索引数据集**
//多种索引，切片方式
dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")
dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]")
dataset = load_dataset("madao33/new-title-chinese", split=["train[:50%]", "train[50%:]"])

- **数据操作：切分、过滤**
dataset["train"][0], dataset["train"].select([0, 1]), dataset["train"]['title']  //选取
sub_data = dataset.train_test_split(test_size=0.1) //切分
sub_data = dataset.train_test_split(test_size=0.1, ***stratify_by_column="label"***) //按label平衡切分
filter_dataset = dataset.filter(lambda example: "中国" in example["title"]) //过滤

- **map**
```
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    # label就是title编码的结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# batched: 加速，num_proc: 线程数
processed_datasets = datasets.map(preprocess_function, batched=True, num_proc=24)
processed_datasets
```

- **with DataCollator**
```
def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_dataset = dataset.map(process_function, batched=True, remove_columns=dataset.column_names)
collator = DataCollatorWithPadding(tokenizer=tokenizer)
dl = DataLoader(tokenized_dataset, batch_size=32, collate_fn=collator, shuffle=True)
```

- **保存和加载**
processed_datasets.save_to_disk("./processed_data")
processed_datasets = load_from_disk("./processed_data")
// 各种加载方法
// 直接加载文件
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = load_dataset("csv", data_dir="./data/", split='train')
// 从pandas加载,也可以用load_dataset指定类型
dataset = Dataset.from_pandas(df)
// 从list加载,也可以用load_dataset指定类型
data = [{"text": "abc"}, {"text": "def"}]
dataset = Dataset.from_list(data)

---

# evaluate
各种任务支持的评价指标，可以在[这里](https://huggingface.co/tasks)找到，进去任务寻找Metrics即可

- **加载和查看**
evaluate.list_evaluation_modules(with_details=True)
accuracy = evaluate.load("accuracy")
accuracy, accuracy.description, accuracy.inputs_description //属性

- **指标计算**
```
# 全局计算
accuracy = evaluate.load("accuracy")
results = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])

# 迭代计算
accuracy = evaluate.load("accuracy")
for ref, pred in zip([0,1,0,1], [1,0,0,1]):
    accuracy.add(references=ref, predictions=pred)
accuracy.compute()

# batch迭代计算
accuracy = evaluate.load("accuracy")
for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
    accuracy.add_batch(references=refs, predictions=preds)
accuracy.compute()

# 多个指标
clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
results = clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
```

- **可视化**
```
from evaluate.visualization import radar_plot

data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]

model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]

plot = radar_plot(data=data, model_names=model_names)
```

---

# trainer
各种任务支持的评价指标，可以在[这里](https://huggingface.co/tasks)找到，进去任务寻找Metrics即可

- **TrainingArguments**
重要参数:
  - output_dir                               # 输出文件夹
  - per_device_train_batch_size    # 训练时的batch_size
  - per_device_eval_batch_size     # 验证时的batch_size
  - logging_steps                          # log 打印的频率
  - evaluation_strategy                 # 评估策略
  - save_strategy                           # 保存策略
  - save_total_limit                        # 最大保存数
  - learning_rate                            # 学习率
  - weight_decay                           # weight_decay
  - metric_for_best_model            # 设定评估指标
  - load_best_model_at_end         # 训练完成后加载最优模型

- **trainer**
trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=evaluate.load("accuracy"))

- **训练和评估**
trainer.train()
trainer.evaluate(tokenized_datasets["test"])
trainer.predict(tokenized_datasets["test"])