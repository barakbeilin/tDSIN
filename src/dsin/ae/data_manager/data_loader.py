from fastai import *
from fastai.vision import *


class ImageSiTuple(ItemBase):
    def __init__(self, img, si_img):
        self.img, self.si_img = img, si_img
        self.obj = (img, si_img),
        self._update_data()
    
    @staticmethod
    def data_to_si_img_and_img(data):
        COLOR_CHANNEL = len(data.shape) - 3
        combined_nof_color_channels = data.shape[COLOR_CHANNEL]

        if combined_nof_color_channels % 2 != 0:
            raise ValueError(f"bad shape in {data.shape}")
        num_of_color_channels = combined_nof_color_channels // 2 
        img = data[:,0: num_of_color_channels,...]
        si_img = data[:,num_of_color_channels:,...]
        return img, si_img


    def _update_data(self):
        if self.img.data.shape == self.si_img.data.shape:
            self.data = torch.cat([self.img.data, self.si_img.data],dim =0)
        else:
            self.data = self.img.data

    def apply_tfms(self, tfms, **kwargs):
        self.img = self.img.apply_tfms(tfms, **kwargs)
        self.si_img = self.si_img.apply_tfms(tfms, **kwargs)
        self._update_data()
        return self

    def to_one(self):
        return Image(torch.cat([self.img.data, self.si_img.data], 1))


class SegmentationProcessor(PreProcessor):
    def __init__(self, ds: ItemList): self.classes = ds.classes
    def process(self, ds: ItemList):  ds.classes, ds.c = self.classes, len(
        self.classes)


class SideinformationImageImageList(ImageList):
    _label_cls = ImageList

    def __init__(self, items, si_items=None, **kwargs):
        """Parameters:
            items - list of images.
            si_items - list of side information images.
        """
        # items passed to the superclass so that when call lbal_from_func the
        # obejct that gets called is `items` so this is the yb that's also pased
        # to the loss function.
        super().__init__(items, **kwargs)
        self.si_items = si_items
        self.copy_new.append('si_items')

    def get(self, i):
        # get an image, opened by self.open()
        img = super().get(i)
        si_img_file_name = self.si_items[i]
        return ImageSiTuple(img=img, si_img=self.open(si_img_file_name))

    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_image(fn, div=True, convert_mode=self.convert_mode)

    def reconstruct(self, t):
        # def reconstruct(self, t, x):
        """Called in data.show_batch(), learn.predict() or learn.show_results()
        Transform tensor back into an ItemBase

        DONT CHANGE x !
        """
        
        return ImageSiTuple(Image(t), Image(t))

    @classmethod
    def from_csv(cls, path: PathOrStr, csv_names: List, header: str = None,
                 delimiter: str = '\n', pct: float = 1.0,
                 **kwargs) -> 'SideinformationImageImageList':
        """
        Get the filenames in `path/csv_name` for each csv_name in csv_names.
        csv file opened with `header` with delimiter.
        """
        path = Path(path)
        dfs = [pd.read_csv(path/csv_name, header=None, delimiter=delimiter)
               for csv_name in csv_names]
        df = pd.concat(dfs, ignore_index=True)

        # keep precentage of number of items, so that the list of items is even
        n_items = max(round(len(df) * pct), 2)
        n_items = n_items if n_items % 2 == 0 else n_items - 1

        # image path's are intermittent
        si_img_df, img_df  = df.iloc[1:n_items:2], df.iloc[0:n_items:2]

        si_items = ImageList.from_df(path=path, df=si_img_df, **kwargs).items
        res = super().from_df(path=path, df=img_df, si_items=si_items, **kwargs)
        return res

    def show_xys(self, xs, ys, figsize: Tuple[int, int] = (12, 6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows, rows, figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int] = None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (12, 3*len(xs)))
        fig, axs = plt.subplots(2, len(xs), figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        sub_figsize = (figsize[0]//3, figsize[1])
        if len(axs.shape) == 1:
            xs[0].to_one().show(ax=axs[0], figsize=sub_figsize, **kwargs)
            zs[0].show(ax=axs[1], figsize=sub_figsize, **kwargs)
            return
        for i, (x, z) in enumerate(zip(xs, zs)):
            x.to_one().show(ax=axs[0, i], figsize=sub_figsize, **kwargs)
            z.show(ax=axs[0, i], figsize=sub_figsize, **kwargs)
