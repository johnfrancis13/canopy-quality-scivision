from canopy_quality import treenet_ms, treenet_rgb
import torch
import numpy as np
from scivision import load_dataset
from scivision import load_pretrained_model

# test numpy array
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"

# load and predict MS model
model = treenet_ms()
image = np.random.randint(255, size=(14, 240, 240), dtype=np.uint8) ## create 14 band image
y = model.predict(image)

# load and predict RGB model
model = treenet_rgb()
images = np.random.randint(255, size=(3, 240, 240), dtype=np.uint8) ## create 3 band image
y = model.predict(images)

# load pretrained model from scivision
model = load_pretrained_model('.scivision/model.yaml', model_selection='treenet_ms')
dataset = load_dataset('.scivision/data.yaml')
img = dataset['canopy_quality'](image_type='ms', image_number=5).read()
y = model.predict(img)