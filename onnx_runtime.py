from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from fastapi.responses import HTMLResponse
import onnx
import onnxruntime as ort
import numpy as np
import torch

app = FastAPI()

# Allow '127.0.0.1' as a valid host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to limit this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

model_path1 = "./ag_news_model64.onnx"
model_path2 = "./ag_news_model128.onnx"

onnx_model1 = onnx.load(model_path1)
onnx.checker.check_model(onnx_model1)
onnx_model2 = onnx.load(model_path2)
onnx.checker.check_model(onnx_model2)



ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."


ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}



class Item(BaseModel):
    text: str = ex_text_str


@app.get("/")
async def predict_category():
    text = torch.tensor(text_pipeline(ex_text_str))
    ort_sess = ort.InferenceSession(model_path1)
    outputs = ort_sess.run(None, {'input': text.numpy(), 'offsets': torch.tensor([0]).numpy()})
    result = outputs[0].argmax(axis=1)+1
    model1_result = f"Model 1 result:- This is a {ag_news_label[result[0]]} news"
    
    ort_sess = ort.InferenceSession(model_path2)
    outputs = ort_sess.run(None, {'input': text.numpy(), 'offsets': torch.tensor([0]).numpy()})
    result = outputs[0].argmax(axis=1)+1
    model2_result = f"Model 2 result:- This is a {ag_news_label[result[0]]} news"
    
    content = f"""
    {model1_result}/n
    {model2_result}
    """
    return {"message":content}
    


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004, log_level="info")


# uvicorn onnx_runtime:app --reload
