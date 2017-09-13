import os

from PIL import Image
import numpy
import six

from chainer.dataset import dataset_mixin

def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


class PreloadedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', dtype=numpy.float32):
        imgs = [_read_image_as_array(os.path.join(root, path), dtype) for path in paths]
        self._imgs = imgs

    def __len__(self):
        return len(self._imgs)

    def get_example(self, i):
        image = self._imgs[i]

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        return image.transpose(2, 0, 1)