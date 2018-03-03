from lib.model import FaceDetectionRegressor
from lib.image import imread, draw_rectangle, imwrite
from lib.utils import merge

# FaceDetectionRegressor
model = FaceDetectionRegressor()

# load weights
model.load_weights('./models')

from os import listdir, path
test_dir = './test/'
result_dir = './results/'
for file in listdir(test_dir):
    image = imread(path.join(test_dir, file))

    predictions = model.predict(image, threshold=.2)
    predictions = merge(predictions)

    print (file, 'detect', len(predictions), 'faces')

    for prediction in predictions:
        draw_rectangle(image, (int(prediction[0]), int(prediction[1])), (int(prediction[2]), int(prediction[3])))
    imwrite(path.join(result_dir, file), image)
