import cv2
import matplotlib.pyplot as plt
import json
def drawing_labels(img_path,annotation_dict):
    img = cv2.imread(img_path)
    for key in annotation_dict["text"].keys():
        label_id = annotation_dict["text"][key]["id"]
        shape = list(annotation_dict["text"][key].keys())[1]
        coordinates_list = annotation_dict["text"][key][shape]
        replacement_text = annotation_dict["text"][key]["replacementText"]
        value = annotation_dict["text"][key]["value"]
        image=cv2.rectangle(img,pt1=coordinates_list[0],pt2=coordinates_list[1],color=(255,0,0),thickness=2)
        
        image=cv2.putText(image,text=value,org=coordinates_list[0],fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
        annotationed_image_path = img_path.replace(".png","_annotationed.png")
        cv2.imwrite(annotationed_image_path, image) 
        # image.save(annotationed_image_path) 
    return annotationed_image_path

def drawing_arrows(img_path:str,annotation_dict:dict):
    img = cv2.imread(img_path)
    
    return annotationed_image_path

def drawing_arrow_heads(img_path:str,annotation_dict:dict):
    img = cv2.imread(img_path)
    for head in list(annotation_dict["arrowHeads"].keys()):
        orientation = annotation_dict["arrowHeads"][head]["orientation"]
        
    return annotationed_image_path

def drawing_blobs(img_path:str,annotation_dict:dict):
    img = cv2.imread(img_path)
    
    return annotationed_image_path
image=drawing_labels("/Users/alexiskaldany/school/CAP22FA/example_data/0.png",json.load(open("/Users/alexiskaldany/school/CAP22FA/example_data/0.png_annotation.json")))

