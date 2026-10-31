"""本のリポジトリ版の代わりに scikit-learn の OpenML から MNIST を取得する。

OpenML の 70,000 件は先頭 60,000 件が訓練、残り 10,000 件がテストで、元の MNIST と同じ分け方になる。
変換済みデータは、このファイルと同じディレクトリの mnist.npz に保存する
（scikit-learn のキャッシュからの読み込みは ARFF の変換に 20 秒ほどかかるため）。
"""

import os

import numpy as np
from sklearn.datasets import fetch_openml


_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")


def _load_arrays():
    if os.path.exists(_CACHE_PATH):
        with np.load(_CACHE_PATH) as cached:
            return tuple(cached[key] for key in ("x_train", "t_train", "x_test", "t_test"))

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    images = np.asarray(mnist.data, dtype=np.uint8)
    labels = np.asarray(mnist.target, dtype=np.uint8)
    arrays = (images[:60000], labels[:60000], images[60000:], labels[60000:])
    np.savez_compressed(
        _CACHE_PATH,
        x_train=arrays[0],
        t_train=arrays[1],
        x_test=arrays[2],
        t_test=arrays[3],
    )
    return arrays


def _change_one_hot_label(labels):
    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST を読み込み、(訓練画像, 訓練ラベル), (テスト画像, テストラベル) を返す。"""
    x_train, t_train, x_test, t_test = _load_arrays()

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)

    if not flatten:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return (x_train, t_train), (x_test, t_test)
