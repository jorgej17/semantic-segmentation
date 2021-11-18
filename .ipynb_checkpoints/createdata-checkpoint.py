import numpy as np
import cv2 as cv
import os

def get_image_patches(source_image, target_image, patch_size):
    """
        This function will get patches to the correct image size
        for the model. I.e. BEiT is pretrained om 224x224 images.
        Therefore we have to create patches of size 224x224
        for each of the 1500x1500 images in the dataset.
    """

    img_h, img_w = source_image.shape[0], source_image.shape[1]
    patch_h, patch_w = patch_size[0], patch_size[1]

    y_patches = img_h // patch_h
    x_patches = img_w // patch_w

    if img_h % patch_h != 0:
        y_patches += 1

    if img_w % patch_w != 0:
        x_patches += 1

    # y, x, patch_size, patch_size, channels ---> [7, 7, 224, 224, 3/1]
    # a = [2, 2, 2]
    source_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 3), dtype=np.uint8)
    target_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 1), dtype=np.uint8)

    for y in range(y_patches):
        for x in range(x_patches):
            source_patch = np.full((patch_size[0], patch_size[1], 3), (255, 102, 255), dtype=np.uint8)
            target_patch = np.zeros((patch_size[0], patch_size[1], 1), dtype=np.uint8)
            #                               0 * 224 = 0 : (0 + 1) * 224 = 224
            temp_source_patch = source_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]
            temp_target_patch = target_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]

            temp_target_patch = np.amax(temp_target_patch, axis=2).reshape(temp_target_patch.shape[0], temp_target_patch.shape[1], 1)

            if temp_source_patch.shape[0] != patch_size[0] or temp_source_patch.shape[1] != patch_size[1]:
                source_patch[:temp_source_patch.shape[0], :temp_source_patch.shape[1]] = temp_source_patch
                target_patch[:temp_target_patch.shape[0], :temp_target_patch.shape[1]] = temp_target_patch
            else:
                source_patch = temp_source_patch
                target_patch = temp_target_patch

            source_patches[y, x] = source_patch
            target_patches[y, x] = target_patch

    source_patches = source_patches.reshape((source_patches.shape[0] * source_patches.shape[1]), source_patches.shape[2], source_patches.shape[3], source_patches.shape[4])
    target_patches = target_patches.reshape((target_patches.shape[0] * target_patches.shape[1]), target_patches.shape[2], target_patches.shape[3], target_patches.shape[4])
    
    target_patches = (target_patches > 0).astype(int)
 
    return source_patches, target_patches


def create_image_tiles_for_custom_dataset(dataset_name, dataset_type, img_path, label_path, tilesize):
    print("Creating images and labels with tilesize", tilesize, "at", "datasets/" + dataset_name)

    main_dir = "/home/jorgej17/semantic-segmentation"

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    dataset_dir = os.path.join(main_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    img_dir, ann_dir = os.path.join(dataset_dir, "img_dir"), os.path.join(dataset_dir, "ann_dir")

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    if not os.path.exists(ann_dir):
        os.mkdir(ann_dir)

    datatype_img_dir, datatype_ann_dir = os.path.join(img_dir, dataset_type), os.path.join(ann_dir, dataset_type)

    if not os.path.exists(datatype_img_dir):
        os.mkdir(datatype_img_dir)

    if not os.path.exists(datatype_ann_dir):
        os.mkdir(datatype_ann_dir)

    img_files = sorted([os.path.join(img_path, file_path) for file_path in os.listdir(img_path)])
    label_files = sorted([os.path.join(label_path, file_path) for file_path in os.listdir(label_path)])

    for img_file_path, label_file_path in zip(img_files, label_files):
        assert img_file_path.split("/")[-1].split(".")[0] == label_file_path.split("/")[-1].split(".")[0]

    print("All images and labels has matching names")

    for img, label in zip(img_files, label_files):
        img_tiles, label_tiles = get_image_patches(cv.imread(img), cv.imread(label), tilesize)
        for i, (img_tile, label_tile) in enumerate(zip(img_tiles, label_tiles)):
            cv.imwrite(os.path.join(datatype_img_dir, img.split("/")[-1].split(".")[0] + "_" + str(i) + "." + img.split("/")[-1].split(".")[1]), img_tile)
            cv.imwrite(os.path.join(datatype_ann_dir, label.split("/")[-1].split(".")[0] + "_" + str(i) + "." + label.split("/")[-1].split(".")[1]), label_tile)

    print("Finished writing images to:", dataset_dir)

if __name__ == "__main__":
    create_image_tiles_for_custom_dataset(dataset_name="data256HLV", dataset_type="val", img_path="data512HLV/img_dir/val", label_path="data512HLV/ann_dir/val", tilesize=(256, 256))