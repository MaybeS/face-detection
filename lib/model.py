from .tfnet import SimpleNet
from .yolo import YOLO

class FaceDetectionRegressor:
  def __init__(self, path):
    self.batch = 32
    self.epoch = 1
    self.name = 'face'
    self.step = 78590
    self.lr = 1e-5
    self.gpu = 0.0
    self.load = False
    self.train = False
    self.save = 20000
    self.keep = 20
    self.path = path
    self.scale = "1,1,.5,5."
    self.load_weights(self.path)

  def fit(self, X, y):
    self.model.train(X, y, self.batch, self.epoch)

  def predict(self, X):
    predictions = self.model.predict(path=X, threshold=0.35, batch=self.batch, merge=True)
    return predictions

  def load_weights(self, weight_path):
    yoloNet = YOLO(weight_path)
    self.model = SimpleNet(yoloNet)
    self.model.setup_meta_ops(self.save, self.lr, self.scale, self.gpu, self.train, self.load, self.keep)

  def save_weights(self, weight_path):
    self.model.save(weight_path)
