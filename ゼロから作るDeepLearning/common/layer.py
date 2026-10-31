# レイヤ（05-back-propagation.ipynb から移したもの）
import numpy as np

from common.activation import softmax
from common.loss import cross_entropy_error


# ReLUレイヤ
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # 例） [-1.0,  2.0, -3.0,  4.0]
        out = x.copy()       # 例） [True, False, True, False]
        out[self.mask] = 0   # 例） [ 0.0, 2.0,  0.0, 4.0]
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# Affineレイヤ
# ニューラルネットワークの順伝播で使う行列の積を、幾何学の分野ではアフィン変換を呼ぶ
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 畳み込み層の出力 (N, C, H, W) のような4次元の入力は (N, C*H*W) に平らにする
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape) # 入力と同じ形に戻す
        return dx


# Softmax-with-Lossレイヤ
# 推論と学習の2つのフェーズがある
#  推論では、最大値にのみ興味があるので、正規化しない出力（スコア）を出すことが多い
#  学習では、正規化する必要があるので、Softmaxレイヤが必要となる

# softmax＋交差エントロピー誤差、恒等関数＋2乗和誤差のような組み合わせだと
# 逆伝播が出力と教師データの差で表現されるようになっている
# ↑こうなるように、損失関数は設計されている
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None    # softmaxの出力
        self.t = None    # 教師データ（one-hot-vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
