from .tfnet import SimpleNet
from .yolo import YOLO

class FaceDetectionRegressor:
  def __init__(self, gpu=0.0):
    self.gpu = gpu
    self.model = None

  def predict(self, X, threshold=0.4, merge=False):
    predictions = self.model.predict(img=X, threshold=threshold, merge=merge)
    return predictions

  def load_weights(self, weight_path='./models'):
    yoloNet = YOLO(weight_path)
    self.model = SimpleNet(yoloNet)
    self.model.setup_meta_ops(self.gpu)