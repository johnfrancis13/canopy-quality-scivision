import numpy as np
from canopy_model.model import treenet


# load model
model = treenet(model_type = "multi_spectral")

# test numpy array
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"

## create 14 band image
image = np.random.randint(255, size=(1, 14, 240, 240), dtype=np.uint8)

##predict
y = model.predict(np_image=image)
