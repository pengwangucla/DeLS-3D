"""
    Data Iter for training
"""

# pylint: skip-file
from collections import OrderedDict,namedtuple

Batch = namedtuple('Batch', ['data'])
import logging
import pdb

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def image_set_2_seqs(image_set, cameras, max_len=None, rand_idx=None):
    """
        Convert the set of images to video sequences based on image name
    """

    def is_conseq_frame(name, last_name):
        record = name.split('/')[0]
        record_id = int(record[-3:])
        record_last = last_name.split('/')[0]
        record_last_id = int(record_last[-3:])
        if record_id in [record_last_id, record_last_id + 1]:
            return True
        return False

    cam_seqs = []
    for Camera in cameras:
        cam_seqs.append([line for line in image_set \
                              if Camera in line])
    res_seqs = []
    for seq in cam_seqs:
        i_last = 0
        for i in range(len(seq)):
            if not is_conseq_frame(seq[i], seq[max(i-1, 0)]):
                res_seqs.append(seq[i_last:i])
                i_last = i
        if len(seq) > i_last:
            res_seqs.append(seq[i_last:len(seq)])


    # split the array with maximum len
    if max_len:
        pass

    return res_seqs



def get_intr_key(keys, filename):
    """get the correct intrinsic for each image.
    """
    intr_key = keys[0]
    found = False
    for key in keys:
        if key in filename:
            intr_key = key
            found = True
            break

    if not found:
        raise ValueError('Image not assigned to a intrinsic')

    return intr_key



