import numpy as np
from canopy_model.model import treenet


# load model
model = treenet(model_type = "multi_spectral")

# test numpy array
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"

## create 14 band image
image = np.random.randint(255, size=(14, 240, 240), dtype=np.uint8)

##predict on random
y = model.predict(np_image=image)

# predict on batch
images = np.random.randint(255, size=(25, 14, 240, 240), dtype=np.uint8)
y = model.predict_batch(images, batch_size=16)

# predict on actual image
a= np.load("../example_data/P_1_X0_1_X1_240_Y0_1_Y1_240_113.npy",allow_pickle=True)
y = model.predict(np_image=a)