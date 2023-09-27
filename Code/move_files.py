DATA_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')


def move_files(source, destination):
    allfiles = os.listdir(source)
    
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        os.rename(src_path, dst_path)

    os.remove(source)


train_images = os.path.join(TRAIN_DIR, 'ribfrac-train-images')
train_images_1 = os.path.join(train_images, 'Part1')
train_images_2 = os.path.join(train_images, 'Part2')

move_files(train_images_1, train_images)
move_files(train_images_2, train_images)

train_labels = os.path.join(TRAIN_DIR, 'ribfrac-train-labels')
train_labels_1 = os.path.join(train_labels, 'Part1')
train_labels_2 = os.path.join(train_labels, 'Part2')

move_files(train_labels_1, train_labels)
move_files(train_labels_2, train_labels)