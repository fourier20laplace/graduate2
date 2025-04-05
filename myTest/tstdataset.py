import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils      import *
channel, im_size, train_n_classes, test_n_classes, dst_train, dst_test = get_dataset(
        "resisc45",
        "/home/lmh/.cache/huggingface/hub/datasets--timm--resisc45",
        zca=0
    )
train_images_all, train_labels_all, train_indices_class = my_organize_dst(
                                                                  dst_train,
                                                                  train_n_classes,
                                                                  print_info=True
                                                              )