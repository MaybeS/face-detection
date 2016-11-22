from .tfnet import SimpleNet
from .yolo import YOLO

class FaceDetectionRegressor:
  def __init__(self, path, gpu=0.0):
    self.gpu = gpu
    self.path = path
    self.load_weights(self.path)

  def predict(self, X, threshold=0.4, merge=False):
    predictions = self.model.predict(img=X, threshold=threshold, merge=merge)
    return predictions

  def load_weights(self, weight_path):
    yoloNet = YOLO(weight_path)
    self.model = SimpleNet(yoloNet)
    self.model.setup_meta_ops(self.gpu)
