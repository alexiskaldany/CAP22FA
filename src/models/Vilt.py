from transformers import ViltProcessor, ViltForQuestionAnswering,Trainer
import requests
from PIL import Image

import json
import pandas as pd
from pathlib import Path
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
DATA_JSON = DATA_DIRECTORY / "question_and_answers_dict_id_key.json"
# prepare inputs
with open(DATA_JSON) as f:
    data = json.load(f)
image_dimensions = (620, 480)
def load_image_resize_convert(image_path):
    image = Image.open(image_path)
    image = image.resize(image_dimensions)
    image = image.convert("RGB")
    return image
# forward pass
def run_model(data):
    questions = []
    correct_answers = []
    predicted_answers = []
    for index,image_id in enumerate(list(data.keys())):
        print(image_id)
        question_list = data[image_id]
        print(data[image_id])
        if len(question_list) > 0:
            image_path = question_list[0]["image_path"]
        if len(question_list) == 0 or question_list == None:
            continue
        image = load_image_resize_convert(image_path)
        if index% 100 == 0:
            print(f"Processing image {index}")
        for q in question_list:
            question = q["question"]
            questions.append(question)
            answer = q["answer"]
            correct_answers.append(answer)
            encoding = processor(image, answer, return_tensors="pt")
            outputs = model(**encoding)
            loss = outputs.loss
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            predicted_answer= model.config.id2label[idx]
            predicted_answers.append(predicted_answer)
        
    return [questions, correct_answers, predicted_answers]

outputs = run_model(data)
df = pd.DataFrame(outputs)
df.to_csv(DATA_DIRECTORY+"vilt.csv")