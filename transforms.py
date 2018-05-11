from skimage.util import random_noise, img_as_float
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

class RandomNoise(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self):
        self.seed=None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to add noise to
        Returns:
            PIL Image: Noisy Image.
        """
        #variance = np.var(img)
        #print("var: ", variance)
        variance=0.5
        return random_noise(img, var=(variance*0.7) )

    def __repr__(self):
        return self.__class__.__name__ + '(seed={0})'.format(self.seed)

class RandomNoiseWithGT(RandomNoise):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to add noise to
        Returns:
            PIL Image: Noisy Image + GT in vertical.
        """
        img_arr = np.asarray(img)

        corrupt = super(RandomNoiseWithGT,self).__call__(img_arr)
        pair = [corrupt, img_as_float(img_arr)]
        #pair = [corrupt, corrupt]

        imgs_comb = np.hstack( i for i in pair )
        imgs_comb = Image.fromarray( imgs_comb)
        #imgs_comb.save( 'Trifecta_vertical.jpg' )
        #print(imgs_comb)

        return imgs_comb