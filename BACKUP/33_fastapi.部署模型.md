# [fastapi 部署模型](https://github.com/iLovEing/notebook/issues/33)

基于http的形式部署，在pytorch框架上运行

---

# server code
```
from transformers import pipeline
import uvicorn
import json
from fastapi import FastAPI, Request
import torch


CUDA_DEVICE_ID = "0"
DEVICE = torch.device(f'cuda: {CUDA_DEVICE_ID}') if torch.cuda.is_available() else torch.device(f'cpu')

app = FastAPI()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@app.post("/")
async def sentiment_analysis(request: Request):
    global sa_pipeline
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象

    answer = {}
    try:
        desc = json_post_list.get('desc')
        result = sa_pipeline(desc)
        answer['label'] = result[0]['label']
        answer['score'] = format(result[0]['score'], '.4f')
    except Exception as e:
        print(f'error: {e}')
        print(f'json obj {json_post_list}')

    if torch.cuda.is_available():
        torch_gc()

    return answer


if __name__ == '__main__':
    sa_pipeline = pipeline(task="sentiment-analysis", device=DEVICE)
    uvicorn.run(app, host='0.0.0.0', port=12345, workers=1)
```