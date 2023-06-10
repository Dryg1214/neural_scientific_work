import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
import torch.multiprocessing

"""
ims = search_images_ddg('grizzly bear')
print(str(len(ims)) + "мишек")

dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
"""

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])

if __name__ == '__main__':
    bear_types = 'grizzly','black','teddy'
    path = Path('bears')

    if not path.exists():
        path.mkdir()
        for o in bear_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_ddg(f'{o} bear')
            download_images(dest, urls=results)

    """
    fns = get_image_files(path)
    failed = verify_images(fns)
    print(failed)
    failed.map(Path.unlink);
    """

    bears = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
    dls = bears.dataloaders(path)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)

    
    bears = bears.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())
    dls = bears.dataloaders(path)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    