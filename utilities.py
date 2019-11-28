import os
import errno
import pydicom
import numpy as np
import cv2
import copy
import math
from sys import exit
import sys
import datetime
import csv, codecs
from decimal import Decimal
from shutil import copyfile
import random
import pickle

# Another utility
def blockPrint(): # Disable printing
    """
    Let print function be disable
    :return:
    """
    sys.stdout = open(os.devnull, 'w')
def enablePrint(): # Restore for printing
    """
    Let print function be enable
    :return:
    """
    sys.stdout = sys.__stdout__
def python_object_dump(obj, filename):
    """
    Dump python object into a file with file name = filename, so that all python data is save in this file
    :param obj:
    :param filename:
    :return:
    """
    file_w = open(filename, "wb")
    pickle.dump(obj, file_w)
    file_w.close()
def python_object_load(filename):
    """
    Some python object's data are saving in the file with file name = filename by python_object_dump().
    For getting this python object's data back, You need to call python_object_load().
    it will load the python object according to the file with file name = filename
    :param filename:
    :return:
        return python object that data is saved in the file with file name = filename
    """
    try:
        file_r = open(filename, "rb")
        obj2 = pickle.load(file_r)
        file_r.close()
    except:
        try:
            file_r.close()
            return None
        except:
            return None
    return obj2
def create_directory_if_not_exists(path):
    """
    Creates 'path' if it does not exist
    If creation fails, an exception will be thrown
    :param path:    the path to ensure it exists
    """
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            print('An error happened trying to create ' + path)
            raise

# FUNCTIONS - Utility of Horizontal Algorithms
def generate_metadata_to_dicom_dict(dicom_dict):
    """
    Generate more metadata for dicom_dict and put data into dicom_dict itself
    :param dicom_dict:
    :return:
        No return value because the generating metadata will add into dicom_dict['metadata']
    """
    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_dicom_dict(dicom_dict)
    metadata = dicom_dict['metadata']
    metadata['view_scope'] = (view_min_y, view_max_y, view_min_x, view_max_x)

    # Figure out global_max_contour_constant_value
    A = dicom_dict['z'][ sorted(dicom_dict['z'].keys())[0] ]['rescale_pixel_array']
    data = A.ravel()
    sorted_data = np.copy(data)
    sorted_data.sort()
    global_max_contour_constant_value = sorted_data[-20] - 100
    metadata['global_max_contour_constant_value'] = global_max_contour_constant_value

    # metadata['view_scope'] = (view_min_y, view_max_y, view_min_x, view_max_x)
    #(contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
def get_contour_xy_mean(cv_contour):
    """
    From the CV's contour, figure out its x,y mean value in pixel unit
    :param cv_contour:
        cv_contour is with the pixel unit. it suppose to be a contour which is CV's algorithm result with dicom CT's pixel_array
    :return:
        return (x,y) mean value. It's also used to represent center of contour.
    """
    rect_info = get_rect_info_from_cv_contour(cv_contour)
    (x_mean, y_mean) = rect_info[2]
    return (x_mean, y_mean)
def get_contour_area_mm2(contour, ps_x, ps_y):
    """
        The function will get the contour with unit of pixel and return the Area with unit of mm^2
        the ps_x and ps_y is help us to knwo how to convert unit from pixel to mm.
    :param contour:
        the contours with the unit of pixel
    :param ps_x:
        PixelSpacing X of CT image
    :param ps_y:
        PixelSpacing Y of CT image
    :return:
        return value of Area of this contour in mm^2 unit
    """
    area_mm2 = cv2.contourArea(contour) * ps_x * ps_y
    return area_mm2
def convert_to_gray_image(pixel_array):
    """
    Convert dicomt CT's pixel_array into gray image
    :param pixel_array:
    :return:
        gray image
    """
    img = np.copy(pixel_array)
    # Convert to float to avoid overflow or underflow losses.
    img_2d = img.astype(float)
    # Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    # Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    return img_2d_scaled
def get_max_contours(A, constant_value=None, ContourRetrievalMode=cv2.RETR_EXTERNAL):
    """
    To get max contours array by referring the max value constant_value
    :param A:
        INPUT: pixel array A
    :param constant_value:
        To set default constant_value or not.
        If there is constant_value, then follow this value to do algorithm.
        If there is no constant_value(mean is None), then figure out the almost max constant value from A
    :param ContourRetrievalMode:
        Contour Retrieval Mode
    :return:
        return array of contours
    """
    constant = None
    if constant_value == None:
        # Algoruthm to find constant value
        data = A.ravel()
        sorted_data = np.copy(data)
        sorted_data.sort()
        constant = sorted_data[-20] - 100
    else:
        constant = constant_value
    filter_img = np.zeros((A.shape[0], A.shape[1], 3), np.uint8)
    filter_img[A <= constant] = (0, 0, 0)
    filter_img[A > constant] = (255, 255, 255)
    gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    return (contours, constant)
def get_max_contours_by_filter_img(A, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL):
    """
    Figure conoutrs of A 2D array by filter_img.
    More reference : https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
    :param A:
        input pixel array A (2D_
    :param filter_img:
        bit mask image
    :param ContourRetrievalMode:
        Contour Retrieval Mode
    :return:
        return contours array
    """
    gray_image = filter_img
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    return contours
def get_contours_from_edge_detection_algo_01(img, filter_img):
    """
    [The algorithm will get contours, with Tree-Sturcture,  according  to filter_img]
    In our algorithm , the filter_img is made by
    cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    More useful information about adaptiveThreshold, you can refer the document here
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

    :param img:
    :param filter_img:
    :return:
    """
    contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_TREE)
    return contours
def get_contours_from_edge_detection_algo_02(img, filter_img):
    """
    [The algorithm will get contours, with NoneTree-Sturcture,  according  to filter_img]
    In our algorithm , the filter_img is made by
    cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    More useful information about adaptiveThreshold, you can refer the document here
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

    :param img:
    :param filter_img:
    :return:
    """

    contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    return contours
def get_contours_from_edge_detection_algo_03(img):
    """
    [The algorithm will get Local maximum contours with Tree-Sturcture]
    Algorithm will compute and return array of contours by maximum constant value
    The algorithm will figure out it maximum constant for img and use this value to find contours  array.
    the return contours array is Tree Structure. It's mean that it may be exist some contour contained by another contour
    :param img:
        INPUT image
    :return:
        Return Tree-Structured contours array
    """
    (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_04(img):
    """
    [The algorithm will get Local maximum contours with NonTree-Sturcture]
    Algorithm will compute and return array of contours by maximum constant value
    The algorithm will figure out it maximum constant for img and use this value to find contours  array.
    the return contours array is NonTree Structure. It's mean there is not contour contained by another contour
    :param img:
        INPUT image
    :return:
        Return NonTree-Structured contours array
    """

    (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_05(img, contour_constant_value):
    """
    [The algorithm will get Global maximum contours with Tree-Sturcture]
    Algorithm will compute and return array of contours by maximum constant value
    The algorithm will figure out it maximum constant for img and use this value to find contours  array.
    the return contours array is Tree Structure. It's mean that it may be exist some contour contained by another contour

    :param img:
        INPUT  image
    :param contour_constant_value:
        We expect this contour_constant_value is the global maximum constant value for
        all CT files in the same dicom_dict.
    :return:
        list of contours
    """

    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value,
                                                           ContourRetrievalMode=cv2.RETR_TREE)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_06(img, contour_constant_value):
    """
    [The algorithm will get Global maximum contours with Non-Tree-Sturcture]
    Algorithm will compute and return array of contours by maximum constant value
    The algorithm will figure out it maximum constant for img and use this value to find contours  array.
    the return contours array is Tree Structure. It's mean that it may be exist some contour contained by another contour

    :param img:
        INPUT  image
    :param contour_constant_value:
        We expect this contour_constant_value is the global maximum constant value for
        all CT files in the same dicom_dict.
    :return:
        list of contours
    """

    # the img should be rescale_pixel_array
    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value,
                                                           ContourRetrievalMode=cv2.RETR_EXTERNAL)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_07(img, contour_constant_value, ps_x, ps_y):
    """
    Because the value of needle is like Lt Ovoid nad Rt Ovoid and the only difference is area size,
    So we get contour by this the value and limit the area size < 10 mm^2.
    Needle area is not easy to get a perfect value's range perfectly,  but find the contour with area smaller than
    10 mm^2 can filter out Lt' ovoid and Rt' Ovoid.

    :param img:
        The input image your want to make algorithm for it
    :param contour_constant_value:
    :param ps_x:
        To figure the area in mm^2 unit, we need to have the PixelSpacing attribute value in CT file, so that we can
        convert the format of pixel into mm unit
    :param ps_y:
        To figure the area in mm^2 unit, we need to have the PixelSpacing attribute value in CT file, so that we can
        convert the format of pixel into mm unit
    :return:
        array of contours that area < 10 mm^2
    """
    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value,
                                                           ContourRetrievalMode=cv2.RETR_EXTERNAL)
    needle_allowed_area_mm2 = 10
    needle_contours = [contour for contour in contours_without_filter if
                       (get_contour_area_mm2(contour, ps_x, ps_y) < needle_allowed_area_mm2)]
    return needle_contours
def generate_output_to_ct_obj(ct_obj):
    """
    Use get_contours_from_edge_detection_algo0* to computing contours data for ct_obj and save data in it
    :param ct_obj:
        INPUT: ct_obj is one of data in dicom_dict to represent CT data
    :return:
        There is no return data because the computing result is adding into ct_obj
        ['output']['contours']['algo*'] -> The data is computed by algo* and the
        contours data format save in mm unit
        ['output']['contours512']['algo*'] -> The data is computed by algo* and the
        contours data format save in pixel unit
        ['output']['contours_infos'] -> More information after computingalgo*, like mean x,y and area size

    """
    out = ct_obj['output']
    rescale_pixel_array = ct_obj['rescale_pixel_array']
    (view_min_y, view_max_y, view_min_x, view_max_x) = ct_obj['dicom_dict']['metadata']['view_scope']
    global_max_contour_constant_value = ct_obj['dicom_dict']['metadata']['global_max_contour_constant_value']

    #view_pixel_array = rescale_pixel_array[view_min_y:view_max_y, view_min_x:view_max_x]
    img = ct_obj['rescale_pixel_array']
    gray_img = convert_to_gray_image(img)
    gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
    img = img[view_min_y: view_max_y, view_min_x:view_max_x]
    rescale_pixel_array = ct_obj['rescale_pixel_array']
    rescale_pixel_array = rescale_pixel_array[view_min_y: view_max_y, view_min_x:view_max_x]
    filter_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    ct_obj['output']['contours'] = {}
    ct_obj['output']['contours']['algo01'] = get_contours_from_edge_detection_algo_01(img, filter_img)
    ct_obj['output']['contours']['algo02'] = get_contours_from_edge_detection_algo_02(img, filter_img)
    ct_obj['output']['contours']['algo03'] = get_contours_from_edge_detection_algo_03(img)
    ct_obj['output']['contours']['algo04'] = get_contours_from_edge_detection_algo_04(img)
    contour_constant_value = ct_obj['dicom_dict']['metadata']['global_max_contour_constant_value']
    ct_obj['output']['contours']['algo05'] = get_contours_from_edge_detection_algo_05(rescale_pixel_array, contour_constant_value)
    ct_obj['output']['contours']['algo06'] = get_contours_from_edge_detection_algo_06(rescale_pixel_array, contour_constant_value)
    ct_obj['output']['contours']['algo07'] = get_contours_from_edge_detection_algo_07(rescale_pixel_array, contour_constant_value, ps_x, ps_y)

    # Process to contours to fit global pixel img
    ct_obj['output']['contours512'] = {}
    for algo_key in sorted(ct_obj['output']['contours'].keys()):
        ct_obj['output']['contours512'][algo_key] =  copy.deepcopy(ct_obj['output']['contours'][algo_key] )
        contours = ct_obj['output']['contours512'][algo_key]
        for contour in contours:
            for [pt] in contour:
                pt[0] = view_min_x + pt[0]
                pt[1] = view_min_y + pt[1]

    # Generate contours infos like x,y mean and area_mm
    ct_obj['output']['contours_infos'] = {}
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    for algo_key in (ct_obj['output']['contours'].keys()):
        contours = ct_obj['output']['contours'][algo_key]
        contours_infos = []
        for contour in contours:
            contours_info = {}
            #contours_infos.append(contours_info)
            (x,y) = get_contour_xy_mean(contour)
            global_x_pixel = x + view_min_x
            global_y_pixel = y + view_min_y
            area_mm2 = get_contour_area_mm2(contour, ps_x, ps_y)
            contours_info['mean'] = [global_x_pixel, global_y_pixel]
            contours_info['area_mm2'] = area_mm2
            contours_info['contour'] = contour
            contours_infos.append(contours_info)
        ct_obj['output']['contours_infos'][algo_key] = contours_infos
    pass

# FUNCTIONS - Utility for Vertical Algorithm
def distance(pt1, pt2):
    """
    The pt1 and pt2 is n-dimension point and distance function will figure out distance of these to points
    :param pt1:
        n-dimension point 1
    :param pt2:
        n-dimension point 2
    :return:
        return floating value that represent the distance between pt1 and pt2
    """
    axis_num = len(pt1)
    # Assume maximum of axis number of pt is 3
    # Because we may save pt[4] as another appended information for algorithm
    if(axis_num > 3):
        axis_num = 3

    sum = 0.0
    for idx in range(axis_num):
        sum = sum + (pt1[idx] - pt2[idx]) ** 2
    ans = math.sqrt(sum)
    return ans
def get_view_scope_by_dicom_dict(dicom_dict):
    """
    If you set View Scope, the computing time of algorithm will be reduced.
    But the side effect is that maybe you don't know what the best view scope for each patient's case.
    There is some patient's case that is not scanned in middle position. So finally, I think return
    512x512 is the best value because pixel_array in CT is 512x512 size.

    :param dicom_dict:
    :return:
    """
    return (0, 512, 0, 512)
def get_minimum_rect_from_contours(contours, padding=2):
    """
    Find the minimum that include all of contour of contours array
    :param contours:
        array of contour
    :param padding:
        Add padding value for return rect
    :return:
        return rect object represent in format of (x_min, x_max, y_min, y_max)
    """
    rect = (x_min, x_max, y_min, y_max) = (0, 0, 0, 0)
    is_first = True
    for contour in contours:
        reshaped_contour = contour.reshape(contour.shape[0], contour.shape[2])
        for pt in reshaped_contour:
            x = pt[0]
            y = pt[1]
            if is_first == True:
                x_min = x
                x_max = x
                y_min = y
                y_max = y
                is_first = False
            else:
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    rect = (x_min, x_max, y_min, y_max)
    return rect
def is_point_in_rect(pt, rect=(0, 0, 0, 0)):
    """
    To detect point is in rect or not
    :param pt:
    :param rect:
    :return:
        return True/False value to mean point is including in rect or not
    """
    (x_min, x_max, y_min, y_max) = rect
    x = pt[0]
    y = pt[1]
    if x >= x_min and x < x_max and y >= y_min and y < y_max:
        return True
    else:
        return False
def is_contour_in_rect(contour, rect=(0, 0, 0, 0)):
    """
    Detect the contour is including in rect or not
    :param contour:
    :param rect:
    :return:
        return boolean value True/False
    """
    (x_min, x_max, y_min, y_max) = rect
    isContourInRect = True
    reshaped_contour = contour.reshape(contour.shape[0], contour.shape[2])
    for pt in reshaped_contour:
        if False == is_point_in_rect(pt, rect):
            isContourInRect = False
            break
    return isContourInRect
def get_rect_info_from_cv_contour(cv_contour):
    """
    To figure out rect object(x_min, x_max, y_min, y_max) , (width,height) and mean x,y value for cv_contour and
    then return these
    :param cv_contour:
    :return:
        return rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]

    """
    i = cv_contour
    con = i.reshape(i.shape[0], i.shape[2])
    x_min = con[:, 0].min()
    x_max = con[:, 0].max()
    x_mean = con[:, 0].mean()
    y_min = con[:, 1].min()
    y_max = con[:, 1].max()
    y_mean = con[:, 1].mean()
    h = y_max - y_min
    w = x_max - x_min
    x_mean = int(x_mean)
    y_mean = int(y_mean)
    rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
    return rect_info
def get_most_closed_pt(src_pt, pts, allowed_distance=100):
    """
    find which is the closed point of src_pt in point array pts.
    But only find the point that distance smaller than allowed_distance
    :param src_pt:
        source point
    :param pts:
        array of points
    :param allowed_distance:
    :return:
        return None if there is not suitable point
        return one point if there is the best matching point in pts
    """
    if pts == None:
        return None
    if pts == []:
        return None
    dst_pt = None
    for pt in pts:
        if distance(src_pt, pt) > allowed_distance:
            # the point , whoes distance with src_pt < allowed_distance, cannot join this loop
            continue

        if dst_pt == None:
            dst_pt = pt
        else:
            if distance(src_pt, pt) < distance(src_pt, dst_pt):
                dst_pt = pt
        pass
    return dst_pt

# FUNCTIONS - Utility of main rp generate function
def get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid):
    """
    [To covnert lt's ovoid, tandem, rt's ovoid from unit of pixel into unit of mm]
    :param dicom_dict:
        INPUT:The dicom_dict to represent the dicom data for specific case of some patient
    :param lt_ovoid:
        INPUT:array of lt_ovoid's point and each point is in format of pixel unit
    :param tandem:
        INPUT:array of teandem's point and each point is in format of pixel unit
    :param rt_ovoid:
        INPUT:array of rt_ovoid's point and each point is in format of pixel unit
    :return:
        return (metric_lt_ovoid, metric_tandem, metric_rt_ovoid). every metric line is
        present the data format in mm unit
    """
    #(metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
    new_lines = []
    for line in [lt_ovoid, tandem, rt_ovoid]:
        new_line = []
        for pt in line:
            z = pt[2]
            ct_obj = dicom_dict['z'][z]
            x = pt[0] * ct_obj['ps_x'] + ct_obj['origin_x']
            y = pt[1] * ct_obj['ps_y'] + ct_obj['origin_y']
            new_line.append([x,y,z])
        new_lines.append(new_line)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = (new_lines[0], new_lines[1], new_lines[2])
    return (metric_lt_ovoid, metric_tandem, metric_rt_ovoid)
def get_metric_needle_lines_representation(dicom_dict, needle_lines):
    """
    Get metric needle lines with representation of mm unit
    :param dicom_dict:
        INPUT:dicom_dict represent dicom data of some case of some patient
    :param needle_lines:
        needle_lines is needle array and every array represent the points for each needle. every points is save in
        pixel unit.
    :return:
        return metric needle lines with representation of mm unit
    """
    metric_needle_lines = []
    for line in needle_lines:
        metric_needle_line = []
        for pt in line:
            z = pt[2]
            ct_obj = dicom_dict['z'][z]
            x = pt[0] * ct_obj['ps_x'] + ct_obj['origin_x']
            y = pt[1] * ct_obj['ps_y'] + ct_obj['origin_y']
            metric_needle_line.append([x, y, z])
        metric_needle_lines.append(metric_needle_line)
    return metric_needle_lines
def get_applicator_rp_line(metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
    """
    The metric line is array of point with mm unit and this function will get the RP line by
    RP's applicator's rule. The RP's applicator's rule is starting from some tip and go every step in fixed length.
    In this case, the RP applicator line will start in first__purpose_distance_mm and then go each step with
    fixed length = each_purpose_distance_mm
    :param metric_line:
        The metric line is array of point with mm unit
    :param first_purpose_distance_mm:
    :param each_purpose_distance_mm:
    :return:
        return the RP applicator line, which is the line information for our to write to RP file
    """
    if (len(metric_line) == 0):
        return []
    # REWRITE get_metric_pt_info_by_travel_distance, so the get_metric_pt, reduct_distance_step and get_metric_pt_info_travel_distance will not be USED
    def get_metric_pt(metric_line, pt_idx, pt_idx_remainder):
        # print('get_metric_pt(metric_line={}, pt_idx={}, pt_idx_remainder={})'.format(metric_line, pt_idx, pt_idx_remainder))
        pt = metric_line[pt_idx].copy()
        try:
            if (pt_idx + 1 >= len(metric_line)):
                end_pt = metric_line[pt_idx]
            else:
                end_pt = metric_line[pt_idx + 1]
        except Exception as e:
            print('Exception in get_metric_pt() of get_applicator_rp_line()')
            print('pt_idx = {}'.format(pt_idx))
            print('pt_idx_remainder = {}'.format(pt_idx_remainder))
            print('metric_line[{}] = {}'.format(pt_idx, metric_line[pt_idx]))
            raise

        for axis_idx in range(3):
            # diff = end_pt[axis_idx] - pt[axis_idx]
            # diff_with_ratio = diff * pt_idx_remainder
            # print('axis_idx = {} ->  diff_with_ratio = {}'.format(axis_idx, diff_with_ratio) )
            pt[axis_idx] += ((end_pt[axis_idx] - pt[axis_idx]) * pt_idx_remainder)
            # pt[axis_idx] = pt[axis_idx] + diff_with_ratio
        return pt
    def reduce_distance_step(metric_line, pt_idx, pt_idx_remainder, dist):
        # reduce dist and move further more step for (pt_idx, pt_idx_remainder)
        # ret_dist = ??  reduce dist into ret_dist
        # Just implement code here , so that the data move a little distance. (mean reduce dist and move more)

        start_pt_idx = pt_idx
        start_pt_idx_remainder = pt_idx_remainder
        start_pt = get_metric_pt(metric_line, start_pt_idx, start_pt_idx_remainder)

        # To figure out what distance we perfer to reduce in this step
        # And the idea is seperate int to two case
        if start_pt_idx < len(metric_line) - 1:
            # CASE: there is a next pt_idx for start_pt_idx
            # In this case, we let end_pt_idx be the next pt_idx of start_pt_idx
            # So it start_pt_idx +1. and don't forget to reset remainder in to zero
            end_pt_idx = start_pt_idx + 1
            end_pt_idx_remainder = 0

        else:
            # CASE there is no any next _pt_idx for start_pt_idx
            # In this case, we let end_pt_idx point to end idx of line and let remainder be in maximum value (1.0)
            # So the end_pt with idx and remainder can represent the most far point in the metric line.
            end_pt_idx = start_pt_idx
            end_pt_idx_remainder = 1

        end_pt = get_metric_pt(metric_line, end_pt_idx, end_pt_idx_remainder)
        max_reducable_dist = distance(start_pt, end_pt)  # max_reducable_dist in this iteration

        # We have start_pt and end_pt , and we have the dist value
        # So we can try to walk from start_pt to some point which belong to [start_pt, end_pt)
        # There are two cases for this walking
        # CASE 1: the end_pt is not enough to walking dist , so just walking to the end_pt
        # CASE 2: the end_pt is enough and we just need to figure where to stop between [start_pt, end_pt)
        # PS: 'is enough' is mean distance will be reduced to zero, so the end of travel is in  [start_pt, end_pt)
        if dist > max_reducable_dist:  # CASE 1 the end_pt is not enough to walking dist
            dist_after_walking = dist - max_reducable_dist
            walking_stop_pt_idx = end_pt_idx
            walking_stop_pt_idx_remainder = end_pt_idx_remainder
            # return (dist, end_pt_idx, end_pt_idx_remainder)
            return (dist_after_walking, walking_stop_pt_idx, walking_stop_pt_idx_remainder)
        else:  # CASE 2 the end_pt is enough, so walking_stop_pt is between [start_pt, end_pt)
            walking_stop_pt_idx = start_pt_idx
            # Figure out walking_stop_pt_idx_remainder
            segment_dist = distance(start_pt, end_pt)
            if (segment_dist == 0):
                # To solve bug of divide zero, Sometimes the segment_dist will be zero
                segment_dist = 0.000000001
            ratio = dist / segment_dist
            walking_stop_pt_idx_remainder = start_pt_idx_remainder + (1 - start_pt_idx_remainder) * ratio
            dist_after_walking = 0
            return (dist_after_walking, walking_stop_pt_idx, walking_stop_pt_idx_remainder)

        pass
        # return (ret_dist, ret_pt_idx, ret_pt_idx_remainder)
    def get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist):
        dist = travel_dist
        count_max = len(metric_line)
        count = 0
        while (True):
            (t_dist, t_pt_idx, t_pt_idx_remainder) = reduce_distance_step(metric_line, pt_idx, pt_idx_remainder, dist)

            if pt_idx == len(metric_line) - 1 and pt_idx_remainder == 1:
                # CASE 0: This is mean the distanced point will out of the line
                print('out of line and remaind unproces distance = ', t_dist)
                t_pt = metric_line[-1].copy()
                return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)
                break
            if t_dist == 0:
                # CASE 1: All distance have been reduced
                t_pt = get_metric_pt(metric_line, t_pt_idx, t_pt_idx_remainder)
                return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)

            count += 1
            if count > count_max:
                # CASE 2: over looping of what we expect. This is case of bug in my source code
                print('The out of counting in loop is happended. this is a bug')
                t_pt = get_metric_pt(metric_line, t_pt_idx, t_pt_idx_remainder)
                return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)
            pt_idx = t_pt_idx
            pt_idx_remainder = t_pt_idx_remainder
            dist = t_dist

    tandem_rp_line = []
    pt_idx = 0
    pt_idx_remainder = 0
    travel_dist = first_purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder, travel_dist)
    tandem_rp_line.append(t_pt)
    for i in range(100):
        travel_dist = each_purpose_distance_mm
        (pt_idx, pt_idx_remainder) = (t_pt_idx, t_pt_idx_remainder)
        (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder,travel_dist)
        if (t_pt == tandem_rp_line[-1]):
            break
        tandem_rp_line.append(t_pt)

    return tandem_rp_line
def get_HR_CTV_min_z(rs_filepath):
    """

    :param rs_filepath:
        the filepath to the RS file
    :return:
        return the minimum z value for HR-CTV ROI in RS file
    """
    # Step 1. Find the ROI Number whose name is HR-CTV in StructureSetROISequence of RS file
    HR_CTV_ROINumber = -1
    rs_fp = pydicom.read_file(rs_filepath)
    for roiSeq in rs_fp.StructureSetROISequence:
        if roiSeq.ROIName == 'HR-CTV':
            #print('ROI_Number = {}'.format(roiSeq.ROINumber))
            HR_CTV_ROINumber = roiSeq.ROINumber
            break
    # Step 2. In ROIContourSequence of RS file, find the idx that index to the item whose is matched to HR-CTV's ROI Number
    hrctv_roicseq_idx = -1
    for c_idx, roiCSeq in enumerate(rs_fp.ROIContourSequence):
        if roiCSeq.ReferencedROINumber == HR_CTV_ROINumber:
            hrctv_roicseq_idx = c_idx
            break
    # Step 3. Each Item in ROIContourSequence is representing the organ or something like applicator.
    # And each ROIContourSequence[hrctv_roicseq_idx].ContourSequence array is present the labels (many contour) data on some speicfic CT slice.
    # ContourSequence[0] is mean the labels data in the CT slice with minimum z.
    # ContourSequence[-1] is mean the labels data in the CT slice with maximum z.

    # Step 4. return minimum z value for RT-CTV ROI
    # ContourData is a array to show [x1,y1,z1, x2,y2,z2, x3,y3,z3 , ... xn,yn,zn]
    # But in the same item of ContourSequence, all z are the same.
    # So for the ContourData array [x1,y1,z1, ...., xn,yn,zn], all z are the same.
    # You can get z value of this slice by ContourData[2]
    # So ContourSequence[0].ContourData[2] is mean the z value of slice of with minimum z.
    min_z = rs_fp.ROIContourSequence[hrctv_roicseq_idx].ContourSequence[0].ContourData[2]
    return min_z
def wrap_to_rp_file(RP_OperatorsName, rs_filepath, tandem_rp_line, out_rp_filepath, lt_ovoid_rp_line, rt_ovoid_rp_line, needle_rp_lines=[], applicator_roi_dict={}):
    """
    :param RP_OperatorsName:
        The name that you want to write into the OperatorName's field in new RP file
    :param rs_filepath:
        The RS file which is together with CT files. Write some field from RS into new RP.
    :param tandem_rp_line:
        The list of tandem's points that format is fit to RP file
    :param out_rp_filepath:
        wrap_to_rp_file() will wrap file to RP file. And the file path of this RP file is  out_rp_filepath.
    :param lt_ovoid_rp_line:
        The list of Lt'Ovoid's points that format is fit to RP file
    :param rt_ovoid_rp_line:
        The list of Rt'Ovoid's points that format is fit to RP file
    :param needle_rp_lines:
        The list of RP needle lines. Each RP line of these lines is a list of specific needle's points that format is fit to RP file
    :param applicator_roi_dict:
        applicator_roi_dict is the mapping of [ROI Name => ROI Number].
        RP file's ChnnelSequence will match to correct ROI Number for each RP line, or the RP file will be failed to import.
    :return:
    """

    print('len(needle_rp_lines)={}'.format(len(needle_rp_lines)))
    # rp_template_filepath = r'RP_Template/Brachy_RP.1.2.246.352.71.5.417454940236.2063186.20191015164204.dcm'
    # rp_template_filepath = r'RP_Template_Brachy_24460566_implant-5_20191113/RP.1.2.246.352.71.5.417454940236.2060926.20191008103753.dcm'
    rp_template_filepath = r'RP_Template_34135696_20191115/RP.1.2.246.352.71.5.417454940236.2077416.20191115161213.dcm'
    def get_new_uid(old_uid='1.2.246.352.71.5.417454940236.2063186.20191015164204', study_date='20190923'):
        uid = old_uid
        def gen_6_random_digits():
            ret_str = ""
            for i in range(6):
                ch = chr(random.randrange(ord('0'), ord('9') + 1))
                ret_str += ch
            return ret_str
        theStudyDate = study_date
        uid_list = uid.split('.')
        uid_list[-1] = theStudyDate + gen_6_random_digits()
        new_uid = '.'.join(uid_list)
        return new_uid

    # Read RS file as input
    rs_fp = pydicom.read_file(rs_filepath)
    # read RP tempalte into rp_fp
    rp_fp = pydicom.read_file(rp_template_filepath)

    rp_fp.OperatorsName = RP_OperatorsName

    rp_fp.FrameOfReferenceUID = rs_fp.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
    rp_fp.ReferencedStructureSetSequence[0].ReferencedSOPClassUID = rs_fp.SOPClassUID
    rp_fp.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = rs_fp.SOPInstanceUID

    directAttrSet = [
        'PhysiciansOfRecord', 'PatientName', 'PatientID',
        'PatientBirthDate', 'PatientBirthTime', 'PatientSex',
        'DeviceSerialNumber', 'SoftwareVersions', 'StudyID',
        'StudyDate', 'StudyTime', 'StudyInstanceUID']
    for attr in directAttrSet:
        #rs_val = getattr(rs_fp, attr)
        #rp_val = getattr(rp_fp, attr)
        #print('attr={}, \n In RS->{} \n In RP->{}'.format(attr, rs_val, rp_val))
        try:
            val = getattr(rs_fp, attr)
            setattr(rp_fp, attr, val)
        except Exception as ex:
            print('Error is happend in for attr in directAttrSet. Sometimes RS file is out of control')
            print(ex)
        #new_rp_val = getattr(rp_fp, attr)
        #print('after update, RP->{}\n'.format(new_rp_val))

    newSeriesInstanceUID = get_new_uid(old_uid=rp_fp.SeriesInstanceUID, study_date=rp_fp.StudyDate)
    newSOPInstanceUID = get_new_uid(old_uid=rp_fp.SOPInstanceUID, study_date=rp_fp.StudyDate)
    rp_fp.SeriesInstanceUID = newSeriesInstanceUID
    rp_fp.SOPInstanceUID = newSOPInstanceUID
    rp_fp.InstanceCreationDate = rp_fp.RTPlanDate = rp_fp.StudyDate = rs_fp.StudyDate
    rp_fp.RTPlanTime = str(float(rs_fp.StudyTime) + 0.001)
    rp_fp.InstanceCreationTime = str(float(rs_fp.InstanceCreationTime) + 0.001)

    # Clean Dose Reference
    rp_fp.DoseReferenceSequence.clear()


    # The template structure for applicator
    # Tandem -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[0]
    # Rt Ovoid -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[1]
    # Lt OVoid -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[2]
    # For each applicator .NumberOfControlPoints is mean number of point
    # For each applicator .BrachyControlPointSequence is mean the array of points


    BCPItemTemplate = copy.deepcopy(rp_fp.ApplicationSetupSequence[0].ChannelSequence[0].BrachyControlPointSequence[0])
    rp_lines = [tandem_rp_line, rt_ovoid_rp_line, lt_ovoid_rp_line]
    rp_lines = rp_lines + needle_rp_lines

    for idx, rp_line in enumerate(rp_lines):
        print('rp_line[{}] = {}'.format(idx, rp_line))


    #TODO rp_Ref_ROI_Numbers need to match to current RS's ROI number of three applicators
    #rp_Ref_ROI_Numbers = [17, 18, 19]
    #rp_Ref_ROI_Numbers = app_roi_num_list

    #enablePrint()
    SortedAppKeys = sorted(applicator_roi_dict.keys())
    app_roi_num_list = []
    print('mapping of [ROI Name => ROI Number]')
    for applicator_roi_name in SortedAppKeys:
        print('{}->{}'.format(applicator_roi_name, applicator_roi_dict[applicator_roi_name]))
        app_roi_num_list.append(applicator_roi_dict[applicator_roi_name])
    print('app_roi_num_list = {}'.format(app_roi_num_list))
    #rp_Ref_ROI_Numbers = sorted(app_roi_num_list, reverse=True)
    rp_Ref_ROI_Numbers = app_roi_num_list
    print('rp_Ref_ROI_Numbers = {}'.format(rp_Ref_ROI_Numbers))
    #blockPrint()
    rp_ControlPointRelativePositions = [3.5, 3.5, 3.5] # After researching, all ControlPointRelativePositions is start in 3.5
    rp_ControlPointRelativePositions = [3.5 for item in app_roi_num_list]

    #enablePrint()
    print('Dr. Wang debug message')
    for idx, rp_line in enumerate(rp_lines):
        print('\nidx={} -> rp_line = ['.format(idx))
        for pt in rp_line:
            print('\t, {}'.format(pt))
    #blockPrint()
    max_idx = 0
    for idx,rp_line in enumerate(rp_lines):
        if (False and  len(needle_rp_lines) == 0):
            enablePrint()
            print('Case without needles')
            if (idx >= 3):
                break
            blockPrint()

        if (False and idx >= 1): #OneTandem
            enablePrint()
            print('Debug importing RP by only tandem')
            rp_fp.ApplicationSetupSequence[0].ChannelSequence = copy.deepcopy(rp_fp.ApplicationSetupSequence[0].ChannelSequence[0:1])
            blockPrint()
            break
        if (idx >= len(rp_Ref_ROI_Numbers)):
            print('the number of rp_line is larger than len(rp_Ref_ROI_Numbers)')
            break
        if (idx >= len(rp_fp.ApplicationSetupSequence[0].ChannelSequence) ):
            print('the number of rp_line is larger than len(rp_fp.ApplicationSetupSequence[0].ChannelSequence)')
            break
        # Change ROINumber of RP_Template_TestData RS into output RP output file
        # Do  I need to fit ROINumber in RS or not? I still have no answer
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].ReferencedROINumber = rp_Ref_ROI_Numbers[idx]
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].NumberOfControlPoints = len(rp_line)
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.clear()
        for pt_idx, pt in enumerate(rp_line):
            BCPPt = copy.deepcopy(BCPItemTemplate)
            BCPPt.ControlPointRelativePosition = rp_ControlPointRelativePositions[idx] + pt_idx * 5
            BCPPt.ControlPoint3DPosition[0] = pt[0]
            BCPPt.ControlPoint3DPosition[1] = pt[1]
            BCPPt.ControlPoint3DPosition[2] = pt[2]
            BCPStartPt = copy.deepcopy(BCPPt)
            BCPEndPt = copy.deepcopy(BCPPt)
            BCPStartPt.ControlPointIndex = 2 * pt_idx
            BCPEndPt.ControlPointIndex = 2 * pt_idx + 1
            rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.append(BCPStartPt)
            rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.append(BCPEndPt)
        max_idx = idx
    rp_fp.ApplicationSetupSequence[0].ChannelSequence = copy.deepcopy(rp_fp.ApplicationSetupSequence[0].ChannelSequence[0:max_idx+1])
    pydicom.write_file(out_rp_filepath, rp_fp)
    pass

