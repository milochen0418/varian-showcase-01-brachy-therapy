
import os
import sys
import pydicom
from decimal import Decimal
from collections import defaultdict
import numpy as np
from skimage.measure import find_contours

import random
import copy

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
DEVICE =  "/gpu:0"



def get_dataset():
    import Mult_Class_Brachy
    # The way that we use dataset is because
    # We want to use the mapping table dataset.class_names[ci] to convert id into class name.
    # for example

    #    self.add_class("Neck", 23, "GTV-N")
    #    self.add_class("Neck", 24, "CTV-L")
    #    self.add_class("Neck", 25, "R't_kidney")
    #    self.add_class("Neck", 26, "L't_kidney")
    #    self.add_class("Neck", 27, "GTV-T")
    #    We can get "GTV-T" class string by dataset.class_names[27]

    # changed by milochen
    # import Mult_Class_Brachy as maskrcnn
    '''
    if "" != "":
        import Mult_Class as maskrcnn
    else:
        import Mult_Class_Brachy as maskrcnn
    '''


    # Load validation dataset
    # changed by milochen
    # dataset = Mult_Class.NeckDataset()
    dataset = Mult_Class_Brachy.NeckDataset()
    # dataset = maskrcnn.NeckDataset()

    CLASS_DIR = os.path.join("datasets_dicom")
    dataset.load_Neck(CLASS_DIR, "val")

    # Must call before using the dataset
    dataset.prepare()
    #print("print dateset")
    #print(dataset)
    return dataset


def get_model():
    #MASKRCNN_MODEL_WEIGHT_FILEPATH = r"C:/Users/Milo/Desktop/Milo/ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
    MASKRCNN_MODEL_WEIGHT_FILEPATH = r"../ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
    import tensorflow as tf
    # changed by milochen
    # import Mult_Class
    import Mult_Class_Brachy
    import mrcnn.model as modellib
    # 讀取檔案的資料夾
    # changed by milochen
    # config = Mult_Class.NeckConfig()
    # config = Mult_Class_Brachy.NeckConfig()
    config = Mult_Class_Brachy.NeckConfig()

    # Override the training configurations with a few
    # changes for inferencing.
    # 對GPU的設計
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    # Create model in inference mode
    # changed by milochen by Sac's suggestion
    # with tf.device(DEVICE):
    #    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Or, load the last model you trained
    # weights_path = model.find_last()
    # weights_path = r"D:\WatchFolder\mask_rcnn_neck_K8s_0030.h5"
    weights_path = MASKRCNN_MODEL_WEIGHT_FILEPATH
    # Load weights
    print('model.load_weight start')
    model.load_weights(weights_path, by_name=True)
    print('model.load_weight end')
    # Test
    model.detect([np.zeros([512, 512, 3])])
    return model

def to_json(class_idsss, masksss, dataset):
    cn_list = []
    dict_value_newT = []
    for i in range(len(class_idsss)):
        mask = masksss[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        dict_value = []
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            dict_value.append(verts)
        dict_value_new = [dict_value[0].tolist()]
        dict_value_newT.append(dict_value_new)
        ci = class_idsss[i]
        cn = dataset.class_names[ci]  # ci is id in class_ids, and cn is mean class name. e.g, ci=27 -> cn=GTV-T
        # The id->class_name mapping is set by Multi_Class.py NeckDataset -> load_Neck -> self.add_class()
        # for example, self.add_class("Neck", 27, "GTV-T")
        cn_list.append(cn)
    dict_json = dict(zip(cn_list, dict_value_newT))
    return dict_json

def AI_process_by_folder(folder ,model_name):
    print("AI_process_by_folder is calling...")
    print("with folder = {} and model_name = {}".format(folder, model_name) )
    # PS: Assume input CT files folder that only put one Study Case for one patient
    # folder check
    if not os.path.isdir(folder):
        print("{} is not exist folder", folder)
        return None

    filelist = []
    for file in os.listdir(folder):
        filepath = "{}\\{}".format(folder, file)
        filelist.append(filepath)
    return AI_process(filelist, model_name)

#def AI_process(filelist, model_name):
def AI_process_get_predict_result(filelist, model_name):
    print("AI_process is calling...")
    print("with model_name = {} and filelist = {}".format(model_name, filelist) )
    # PS: Assume input CT files folder that only put one Study Case for one patient
    # filelist check
    if filelist == None:
        print("filelist is None")
        return None
    if len(filelist) == 0:
        print("filelist is Empty")
        return None

    # model_name check
    if model_name != "MRCNN_Brachy":
        print("model_name is not correct, please try MRCNN_Brachy")
        return None

        # Process AI Case of MRCNN_Brachy
    return MRCNN_Brachy_AI_process(filelist)

def MRCNN_Brachy_AI_process(filelist):
    print("MRCNN_Brachy_AI_process is calling with filelist = {}".format(filelist) )
    dataset = get_dataset()
    model = get_model()
    label_id_mask = get_label_id_mask(dataset, model, filelist)
    return label_id_mask

def get_label_id_mask(dataset, model, ct_filelist):
    print("get_label_id_mask is calling")

    label_id_mask = defaultdict(dict)
    for filepath in ct_filelist:
        # filepath is a absolute filepath for some ct file
        print(filepath)

        ct_fp = pydicom.read_file(filepath)
        image = ct_fp.pixel_array
        tmp_image = np.zeros([512, 512, 3])
        for ii in range(512):
            for jj in range(512):
                for kk in range(3):
                    tmp_image[ii][jj][kk] = image[ii][jj]
        image = tmp_image
        results = model.detect([image])
        r = results[0]
        mask = to_json(r['class_ids'], r['masks'], dataset)

        x_spacing, y_spacing = float(ct_fp.PixelSpacing[0]), float(ct_fp.PixelSpacing[1])
        origin_x, origin_y, _ = ct_fp.ImagePositionPatient

        for k, v in mask.items():
            pixel_coords = list()
            for x, y in mask[k][0]:
                # pixel_coords.append(float(Decimal(str((x + origin_x - 7) * x_spacing)).quantize(Decimal('0.00'))))
                # pixel_coords.append(float(Decimal(str((y + origin_y - 7) * y_spacing)).quantize(Decimal('0.00'))))
                tmpX = x * x_spacing + origin_x
                tmpY = y * y_spacing + origin_y
                theX = float(Decimal(str(tmpX)).quantize(Decimal('0.00')))  # Some format transfer stuff
                theY = float(Decimal(str(tmpY)).quantize(Decimal('0.00')))  # Some format transfer stuff
                pixel_coords.append(theX)
                pixel_coords.append(theY)

                pixel_coords.append(float(Decimal(str(ct_fp.SliceLocation)).quantize(Decimal('0.00'))))
            label_id_mask[k][ct_fp.SOPInstanceUID] = pixel_coords

    return label_id_mask



def test_case_1():
    model_name = "MRCNN_Brachy"
    ct_folder = "TestCase_Input_CtFolder"
    ret = AI_process_by_folder(ct_folder, model_name)
    print("test_case_1() done")
    #print(ret)



if __name__ == "__main__":
    test_case_1()
