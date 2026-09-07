import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", dest="models")
parser.add_argument("--vision-tower", default="./base_models/vision_tower-clip336")
args = parser.parse_args()
image_checkpoint_info = {  
    "image_checkpoint": {  
        "_target_": "llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower",  
        "download_method_for_image_model": "local",  
        "image_model": str(Path(args.vision_tower).resolve())
    }  
}  

json_file_path_list = args.models or ['./base_models/llava-v1.5-7b/config.json',
                                     './base_models/llava-v1.5-13b/config.json']

for json_file_path in json_file_path_list:
    with open(json_file_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)  
    data.update(image_checkpoint_info)  
    with open(json_file_path, 'w', encoding='utf-8') as file:  
        json.dump(data, file, indent=4, ensure_ascii=False)  
    print(f"Data has been updated with 'image_checkpoint'.")
