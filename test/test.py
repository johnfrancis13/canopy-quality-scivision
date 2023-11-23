import numpy as np
from canopy_model.model import treenet


# load model
model = treenet()

# test numpy array
## create RGB image
image = np.random.randint(255, size=(240, 240, 3), dtype=np.uint8)

##predict
y = model.predict(image)

# test batch
##predict
#image_batch = 
y = model.predict_batch(image_batch, batch_size=16)