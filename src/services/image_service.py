import numpy as np
from PIL import Image


class ImageService:

    def __init__(self, fp):
        self.image = Image.open(fp=fp)
