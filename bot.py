import os
import torch
import cv2
import yaml
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords


weights_path = 'name.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())


image_folder = '&//image.path'
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')  # user desktop path
output_folder = os.path.join(desktop_path, 'runs')  #output folder path
label_folder = 'labels'
img_output_subfolder = 'img'
img_size = 640
dataset = LoadImages(image_folder, img_size=img_size)


model.eval()


os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, label_folder), exist_ok=True)
os.makedirs(os.path.join(output_folder, img_output_subfolder), exist_ok=True)


class_names = model.names  
data_yaml = {
    'train': 'path/to/train', 
    'val': 'path/to/val',      
    'nc': len(class_names),    
    'names': class_names       
}


yaml_path = os.path.join(output_folder, 'data.yaml')
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)
print(f"data.yaml creadted succesfully : {yaml_path}")


for path, img, im0s, _ in dataset:
    filename = os.path.splitext(os.path.basename(path))[0]  

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    
    pred = model(img)[0]

    
    pred = non_max_suppression(pred, 0.25, 0.45)

    
    with open(os.path.join(output_folder, label_folder, f'{filename}.txt'), 'w') as f:
        for j, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:
                    
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / im0s.shape[1]
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / im0s.shape[0]
                    width = (xyxy[2] - xyxy[0]) / im0s.shape[1]
                    height = (xyxy[3] - xyxy[1]) / im0s.shape[0]
                    
                    line = f"{int(cls)} {x_center:.7f} {y_center:.7f} {width:.7f} {height:.7f}\n"
                    f.write(line)

    
    resized_img = cv2.resize(im0s, (640, 640))
    img_output_path = os.path.join(output_folder, img_output_subfolder, f'{filename}.jpg')
    cv2.imwrite(img_output_path, resized_img)

cv2.destroyAllWindows()
