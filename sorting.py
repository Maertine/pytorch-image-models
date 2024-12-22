import os
import shutil

val_destination = '/home/gratzerm/datasets/imagenet/val/'

train_images = os.listdir(val_destination)
for image in train_images:

    l = list(image)[-1]
    if l == 'G':

        split = image.split('_')
        split = split[3].split('.')

        cls_name = split[0]

        if not os.path.exists(val_destination + cls_name):
            print('creating dir: ', 'val/' + cls_name)
            os.makedirs(val_destination + cls_name, exist_ok=True)

        src = val_destination + image
        destination = val_destination + cls_name + '/'
            #print('moving')
            #print(src)
            #print(destination)
        shutil.move(src, destination)