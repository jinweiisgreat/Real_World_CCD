# hugging face
# from PIL import Image
# import requests
#
# import torch
# from transformers import CLIPModel, CLIPProcessor
# model = CLIPModel.from_pretrained("/home/ps/_jinwei/CLIP_L14")
# processor = CLIPProcessor.from_pretrained("/home/ps/_jinwei/CLIP_L14")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
#
# print(probs)  # tensor([[0.2689, 0.7311]], grad_fn=<SoftmaxBackward0>)
# print(f"Image embeddings shape: {outputs.image_embeds.shape}")

# official
import clip

# 列出所有可用的预训练模型
available_models = clip.available_models()
print("可用的CLIP模型:")
for model in available_models:
    print(f"  - {model}")