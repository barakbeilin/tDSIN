from fastai.vision import ImageDataBunch
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=24);
