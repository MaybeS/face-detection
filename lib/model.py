from .tfnet import SimpleNet
from .yolo import YOLO

class FaceDetectionRegressor:
  def __init__(self, weight=None, gpu=0.0):
    self.gpu = gpu
    self.model = SimpleNet(YOLO(weight)) if weight else None

  def predict(self, img, threshold=0.4):
    return self.model.predict(img=img, threshold=threshold)

  def load_weights(self, weight_path='./models'):
    self.model = SimpleNet(YOLO(weight_path))
    self.model.setup_meta_ops(self.gpu)