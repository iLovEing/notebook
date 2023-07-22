# [Hugging Face: transformers](https://github.com/iLovEing/notebook/issues/21)

# pipline

- 基本使用
pipe = pipeline("text-classification"， device="cuda")
pipe("i love u")


- 加载本地任务
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


- 查看支持任务
pipelines.SUPPORTED_TASKS


- 查看任务参数
打印pipe，可以看到具体函数，然后打印函数__doc__ or 直接进入。