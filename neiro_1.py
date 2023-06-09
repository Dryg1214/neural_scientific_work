from fastai.vision.all import *
from fastbook import *
import os
import matplotlib.pyplot as plt

def downloadsameimages(searches: list, path: Path, count: int):
    if count > 200: count=200
    data_folders = list()
    if path.exists():
        for i in searches:
            dest = Path(f'{path}\\{i}')
            if not dest.exists():
                mkdir(dest)
            result = search_images_ddg(f'{i} photo')
            download_images(dest, urls=result[0:count], preserve_filename=True)
            resize_image(dest, max_size=400, dest=dest)
            data_folders += [dest]
    return data_folders



def deletefailedimg(folders:list):
    for i in folders:
        failed = verify_images(get_image_files(i))
        failed.map(os.remove)

if __name__ == '__main__':
    """
    searches = 'owl', 'eagle'
    path = Path('C:\\Users\\Dungeon Master\\neiro_nauch\\data_sets')
    if not path.exists():
        mkdir(path)
    new_folders = downloadsameimages(searches, path, 200)
    deletefailedimg(new_folders)

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path)
    dls.train.show_batch(max_n=6, nrows=2)
    plt.show()
    home = Path('C:\\Users\\Dungeon Master\\neiro_nauch\\')
    classified_object = Path('owl.jpg')
    testfolder = downloadsameimages(['owl'], home, 5)
    deletefailedimg(testfolder)

    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    """

    learn = load_learner('models\\owl_eagle.pkl', cpu=False)
    path = os.listdir(path='owl\\')
    for i, name in enumerate(path):
        #pil_img = Image.open('bird\\bird.jpg') 
        is_classified_object, _, probs = learn.predict(f'owl\\{name}')
        
        print(f'{name} is a: {is_classified_object}')
        print(f'Probability is a owl: {probs[1]:.4f}')
    learn.export('models\\owl_eagle.pkl')
    

    