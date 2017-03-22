from lib.model import FaceDetectionRegressor

# FaceDetectionRegressor
model = FaceDetectionRegressor()

# load weights
model.load_weights('./models')

# prediction
# >> X:           image_path
# >> threshold:   default = 0.4
# >> merge:       default = False, merge intersection boxes
predictions = model.predict('./data/test.jpg', merge=True)
print predictions

from lib.drawer import draw
draw('./data/test.jpg', './results/test.jpg', predictions)
