# from pathlib import Path
# from src.utils.pre_process import *
from transformers import BertTokenizer,DeiTFeatureExtractor,pipeline,ViltForQuestionAnswering
""" 
Setup
"""
# JSON_PATH = DATA_DIRECTORY /"question_and_answers_dict_id_key.json"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# visual_feature_extractor = DeiTFeatureExtractor.from_pretrained(
#     "facebook/deit-base-distilled-patch16-224"
# )
pipe = pipeline("visual-question-answering")

# model= ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
""" 
for key in key_dict:
    num_of_questions = len(key)
    visual_embeddings = 
"""
image = "example_data/0.png"
question = "What is A in the diagram?"
actual_answer = "face"
answer=pipe(image=image,question=question,top_k=50)
print(answer)