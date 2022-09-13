# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import BertTokenizer, VisualBertForMultipleChoice,AutoFeatureExtractor,VisualBertModel
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
import torch
from PIL import Image
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vcr")
max_length = 512
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
visual_feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

image = Image.open("/Users/alexiskaldany/school/CAP22FA/example_data/happy-hungry-man-eating-pizza-using-fork-knife-italian-restaurant-hungry-man-eating-pizza-using-fork-knife-italian-174038619.png")
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_transforms = Compose(
    [RandomResizedCrop(feature_extractor.size), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize]
)
# (batch_size, num_choices, visual_seq_length, visual_embedding_dim)

visual_features = visual_feature_extractor(image,return_tensors="pt",padding="max_length",max_length=512)["pixel_values"]
print(f"Shape of visual_features: {visual_features.shape}")
# visual_embeds = visual_features.expand(1,2,*visual_features.shape[1:])
visual_embeds = torch.nn.functional.interpolate(input=visual_features, size = (512,512))
visual_embeds = visual_embeds
print(f"Shape of visual_embeds: {visual_embeds.shape}")
visual_embeds = visual_features.reshape()
print(f"Shape of visual_embeds: {visual_embeds.shape}")
visual_token_type_ids = torch.ones((1,512), dtype=torch.long)
visual_attention_mask = torch.ones((1,512), dtype=torch.float)
# Labels = index of answer in question-possible answers pairs
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
print(f"This is the labels: {labels}")
encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)
# batch size is 1
# inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
inputs_dict = tokenizer(prompt,return_tensors="pt")
inputs_dict.update(
    {
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "visual_token_type_ids": visual_token_type_ids,
        
    }
)
# print(f"key names: {print(inputs_dict.keys())}")
# print(f"value type: {[type(v) for v in inputs_dict.values()]}")
# print(f"This is input_ids: {inputs_dict['input_ids']}")
# print(f"Shapes: {[inputs_dict[x].shape for x in inputs_dict.keys()]} ")

keys = [x for x in inputs_dict.keys()]
shapes =[inputs_dict[x].shape for x in inputs_dict.keys()]
key_shape = [(k,v) for k,v in zip(keys,shapes)]
# input_list = [inputs_dict[x] for x in inputs_dict.keys()]
# print(input_list)
print(key_shape)
labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)
outputs = model(**inputs_dict)

# loss = outputs.loss
# logits = outputs.logits