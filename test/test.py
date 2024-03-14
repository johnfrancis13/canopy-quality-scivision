from canopy_quality import treenet_ms, treenet_rgb
import torch
import numpy as np
from scivision import load_dataset
from scivision import load_pretrained_model

# test numpy array
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = treenet_ms()

##predict on random image
image = np.random.randint(255, size=(14, 240, 240), dtype=np.uint8) ## create 14 band image
y = model.predict(np_image=image)

# predict on batch with rgb
model = treenet_rgb()

images = np.random.randint(255, size=(25, 3, 240, 240), dtype=np.uint8) ## create 3 band image
y = model.predict_batch(images, batch_size=16)

# predict on actual numpy image
model = treenet_ms()
a= np.load("example_data/numpy/ms/P_1_X0_1_X1_240_Y0_1_Y1_240_113.npy",allow_pickle=True)
a = a[:14]
print(a.shape)
y = model.predict(np_image=a)

# predict on actual geotiff image using scivision load_dataset
model = treenet_ms()
dataset = load_dataset('.scivision/data.yaml')
img = dataset['canopy_quality'](image_type='ms', image_number=5).read()
y = model.predict(img)

# load pretrained model
model = load_pretrained_model('.scivision/model.yaml')