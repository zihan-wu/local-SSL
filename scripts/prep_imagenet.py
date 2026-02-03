import os

def create_val_img_folder(dataset_dir):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print('Created folder {}'.format(newpath))
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))
        else: 
            raise ValueError('Image does not exist in {}'.format(os.path.join(img_dir, img)))

if __name__ == '__main__':
    dataset_dir = '/lcncluster/zihan/datasets/tiny-imagenet-200'
    if os.path.exists(os.path.join(dataset_dir, 'val', 'images')):
        print('Formatting validation image folders for ImageNet')
        create_val_img_folder(dataset_dir)
    else:
        print('Validation image folders do not exist')
