# Try to seperate program into clear verion and useful functions
import os
from pathlib import Path, PureWindowsPath
import errno
import pydicom
import numpy as np
import cv2
import time
import copy
import math
from sys import exit
import sys
import datetime
from IPython.display import display, HTML
import openpyxl
import csv, codecs
from decimal import Decimal
from shutil import copyfile
import random
import pickle

from utilities import blockPrint
from utilities import enablePrint

# FUNCTIONS - Horizontal Algorithm for each CT slice. Use OpenCV to make contours
def get_dicom_dict(folder):
    """

    :param folder:
        There are at least RS and CT file in this folder
        get_dicom_dict() will collect all filepath information about CT, RS, RD and RP.
        it will also get basic information from CT and RS files

    :return:
        return the dicom_dict object.
        From dicom_dict object, you can use all dicom file's information you need for this project.
        So, your process don't need to waste time to reopen again.

        dicom_dict support the following field
        ['z'][-32] will get ct object with z = -32
        Each ct object, which is a dicionary object,  is not also represent a CT file but filepath and dicom_dict.
        ['pathinfo']['rs_filepath'] will get RS filepath. So do rd_filepath, rp_filepath and ct_filelist.
        ['metadata'] save all metadata that represent all folder. for example
        ['metadata']['RS_StudyDate'] is mean RS file's StudyDate
        ['metadata']['RS_PatientID'] is mean RS file's PatientID
        ['metadata']['RS_SOPInstanceUID'] is mean RS file's SOPInstanceUID
        ['metadata']['applicator_roi_dict'] is a dictionary that mapping data from ROIName to ROINumber. This information is colleted  from RS file and
        we need to use this information to write RP file.

    """
    def get_dicom_folder_pathinfo(folder):
        dicom_folder = {}
        ct_filelist = []
        rs_filepath = None
        rd_filepath = None
        rp_filepath = None
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            try:
                ct_dicom = pydicom.read_file(filepath)
                """
                CT Computed Tomography
                RTDOSE Radiotherapy Dose
                RTPLAN Radiotherapy Plan
                RTSTRUCT Radiotherapy Structure Set
                """
                m = ct_dicom.Modality
                if (m == 'CT'):
                    ct_filelist.append(filepath)
                elif (m == 'RTDOSE'):
                    rd_filepath = filepath
                elif (m == 'RTSTRUCT'):
                    rs_filepath = filepath
                elif (m == 'RTPLAN'):
                    rp_filepath = filepath
            except Exception as e:
                pass
        dicom_folder['ct_filelist'] = ct_filelist
        dicom_folder['rs_filepath'] = rs_filepath
        dicom_folder['rd_filepath'] = rd_filepath
        dicom_folder['rp_filepath'] = rp_filepath
        return dicom_folder
    z_map = {}
    ct_filepath_map = {}
    out_dict = {}
    out_dict['z'] = z_map
    out_dict['ct_filepath'] = ct_filepath_map
    out_dict['metadata'] = {}
    out_dict['metadata']['folder'] = folder
    pathinfo = get_dicom_folder_pathinfo(folder)
    out_dict['pathinfo'] = pathinfo

    rs_filepath = out_dict['pathinfo']['rs_filepath']
    rs_fp = pydicom.read_file(rs_filepath)
    out_dict['metadata']['RS_StudyDate'] = rs_fp.StudyDate
    out_dict['metadata']['RS_PatientID'] = rs_fp.PatientID
    out_dict['metadata']['RS_SOPInstanceUID'] = rs_fp.SOPInstanceUID

    # Set metadata for ROINumber list (for wrap rp data)
    rs_fp = pydicom.read_file(rs_filepath)
    if (rs_fp != None):

        # applicator_target_list = ['Applicator1', 'Applicator2', 'Applicator3']
        # Process to get applicator_target_list
        applicator_target_list = []
        for item in rs_fp.StructureSetROISequence:
            if 'Applicator' in item.ROIName:
                applicator_target_list.append(item.ROIName)


        applicator_roi_dict = {}
        for app_name in applicator_target_list:
            for item in rs_fp.StructureSetROISequence:
                if (item.ROIName == app_name):
                    applicator_roi_dict[app_name] = item.ROINumber
                    break
        #print('\napplicator_roi_dict = {}'.format(applicator_roi_dict))
        #display(applicator_roi_dict)
        #print(applicator_roi_dict.values())
        roi_num_list = [int(num) for num in applicator_roi_dict.values()]
        #print(roi_num_list)
        out_dict['metadata']['applicator123_roi_numbers'] = roi_num_list.copy()
        out_dict['metadata']['applicator_roi_dict'] = applicator_roi_dict

    ct_filelist = pathinfo['ct_filelist']
    for ct_filepath in ct_filelist:
        # print('ct_filepath = ', ct_filepath)
        ct_fp = pydicom.read_file(ct_filepath)
        ct_obj = {}
        ct_obj['dicom_dict'] = out_dict
        ct_obj['filepath'] = ct_filepath
        ct_obj["pixel_array"] = copy.deepcopy(ct_fp.pixel_array)
        ct_obj["RescaleSlope"] = ct_fp.RescaleSlope
        ct_obj["RescaleIntercept"] = ct_fp.RescaleIntercept
        ct_obj["rescale_pixel_array"] = ct_fp.pixel_array * ct_fp.RescaleSlope + ct_fp.RescaleIntercept
        ct_obj['ps_x'] = ct_fp.PixelSpacing[0]
        ct_obj['ps_y'] = ct_fp.PixelSpacing[1]
        ct_obj['origin_x'] = ct_fp.ImagePositionPatient[0]
        ct_obj['origin_y'] = ct_fp.ImagePositionPatient[1]
        ct_obj['origin_z'] = ct_fp.ImagePositionPatient[2]
        ct_obj['SliceLocation'] = ct_fp.SliceLocation
        ct_obj['output'] = {} # put your output result in here

        z = ct_obj['SliceLocation']
        z_map[ z ] = ct_obj
        ct_filepath_map[ct_filepath] = ct_obj
        #print('ct_obj={}'.format(ct_obj))
    return out_dict
def generate_output_to_dicom_dict(dicom_dict):
    """
    After call get_dicom_dict(folder), you can get dicom_dict.
    generate_output_to_dicom_dict(dicom_dict) will compute many kind of openCV function and save the temporary data info dicom_dict
    Horizontal Algorithm material will finish after calling this function.
    :param dicom_dict:
        The dicom_dict is INPUT that from the return value of  get_dicom_dict(folder)
    :return:
        No return value but the append more algorithm computing result into dicom_dict
    """

    # generate_output_to_ct_obj will compute any Open CV algorihtm for each CT and save the computing result into dicom_dict for them
    from utilities import generate_output_to_ct_obj
    folder = dicom_dict['metadata']['folder']
    z_map = dicom_dict['z']
    for z_idx, z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        #print('z={}, {}'.format(z, ct_obj.keys()))
        generate_output_to_ct_obj(ct_obj)
        # information is in ct_obj['output']

# FUNCTIONS - Algorithm processing Fucntions
def algo_to_get_pixel_lines(dicom_dict, needle_lines = []):
    """
    After horizontal algorithm have figured out by Horizontal Algorithm,
    algo_to_get_pixel_lines() ia processing in vertical for every ct slice in dicom_dict.
    This algorithm will figure out lt'ovoid, tandem and rt'ovoid in line with unit of pixel
    The needle_lines is a referenced input the help tandem line can more be precise.

    :param dicom_dict:
    :param needle_lines:
    :return:
        (lt_ovoid, tandem, rt_ovoid). And each of it is array of points and each point is unit of pixel
    """
    from utilities import get_rect_info_from_cv_contour
    from utilities import get_minimum_rect_from_contours
    from utilities import is_contour_in_rect

    # type: (dicom_dict) -> (lt_ovoid, tandem, rt_ovoid)
    # Step 1. Use algo05 to get center point of inner contour
    last_z_in_step1 = sorted(dicom_dict['z'].keys())[0]
    center_pts_dict = {} # The following loop will use algo03 to figure L't Ovoid, R't Ovoid and half tandem
    for z in sorted(dicom_dict['z'].keys()):
        #contours = dicom_dict['z'][z]['output']['contours512']['algo03']
        contours = dicom_dict['z'][z]['output']['contours512']['algo05']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo03')
        #plot_with_contours(dicom_dict, z=z, algo_key='algo05')
        # Step 1.1 The process to collect the contour which is inner of some contour into inner_contours[]
        inner_contours = []
        for inner_idx, inner_contour in enumerate(contours):
            is_inner = False
            for outer_idx, outer_contour in enumerate(contours):
                if inner_idx == outer_idx: # ignore the same to compare itself
                    continue
                outer_rect = get_minimum_rect_from_contours([outer_contour])
                if is_contour_in_rect(inner_contour, outer_rect):
                    is_inner = True
                    break
            if (is_inner == True) :
                inner_contours.append(inner_contour)
        # Step 1.2 if there is no any inner contour, then last_z_in_step1 = z and exit the z loop
        if (len(inner_contours)) == 0:
            last_z_in_step1 = z
            break

        # Step 1.3 figure out center point of contour in inner_contour and sorting it by the order x
        print('z = {}, len(inner_contours) = {}'.format(z, len(inner_contours)))
        inner_cen_pts = []
        for contour in inner_contours:
            #rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            inner_cen_pts.append(cen_pt)
        inner_cen_pts.sort(key=lambda pt:pt[0])
        print('z = {}, inner_cen_pts = {}'.format(z, inner_cen_pts) )
        center_pts_dict[z] = inner_cen_pts


    # Step 2. Figure L't Ovoid
    print('STEP 2.')
    lt_ovoid = []
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of L't Ovoid
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    print('sorted(center_pts_dict.keys()) = {}'.format(sorted(center_pts_dict.keys())))
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            lt_ovoid.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        x_mm = center_pts_dict[z][0][0] * ps_x
        y_mm = center_pts_dict[z][0][1] * ps_y
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            lt_ovoid.append(prev_pt)
            print('lt_ovoid = {}'.format(lt_ovoid))
        else:
            break

    # Step 3. Figure R't Ovoid
    rt_ovoid = []
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of R't Ovoid
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    print('sorted(center_pts_dict.keys()) = {}'.format(sorted(center_pts_dict.keys())))
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            prev_pt = ( center_pts_dict[z][-1][0], center_pts_dict[z][-1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            rt_ovoid.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        x_mm = center_pts_dict[z][-1][0] * ps_x
        y_mm = center_pts_dict[z][-1][1] * ps_y
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][-1][0], center_pts_dict[z][-1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            rt_ovoid.append(prev_pt)
            print('rt_ovoid = {}'.format(rt_ovoid))
        else:
            #distance_mm = math.sqrt((x_mm - prev_x_mm) ** 2 + (y_mm - prev_y_mm) ** 2)
            #print('distnace_mm = {}, '.format(distance_mm))

            break

    # Step 4. Figure Tandem bottom-half (thicker pipe part of tandem)
    tandem = []
    #allowed_distance_mm = 4.5 # allowed distance when trace from bottom to tips
    allowed_distance_mm = 4.5  # allowed distance when trace from bottom to tips
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            # It is possible that thicker pipe part of tandem is not scanned in CT file, so that only can detect two pipe in this case.
            # So that when center_pts_dict < 3 in following case after using algo03
            if (len(center_pts_dict[z]) < 3)  :
                print('Bottom-half Tandem say break in loop with z = {}'.format(z))
                break
            prev_pt = ( center_pts_dict[z][1][0], center_pts_dict[z][1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            tandem.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        print('aa and idx_z = ',idx_z, flush= True)
        if ( len(center_pts_dict[z]) <= 1 ):
            # to prevent out of range of list
            continue
        x_mm = center_pts_dict[z][1][0] * ps_x
        y_mm = center_pts_dict[z][1][1] * ps_y
        #print('x_mm = {}, y_mm ={}'.format(x_mm, y_mm))
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][1][0], center_pts_dict[z][1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            tandem.append(prev_pt)
            print('tandem = {}'.format(tandem))
        else:
            break
    #
    # Step 5. The case to process the tandem without thicker pipe in scanned CT. when tandem = [] (empty list)
    if len(tandem) == 0:
        # Step 5.1 find out inner_cotnour of tandem by algo01
        contours = dicom_dict['z'][z]['output']['contours512']['algo01']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo03')
        # Step 5.1.1 The process to collect the contour which is inner of some contour into inner_contours[]
        inner_contours = []
        for inner_idx, inner_contour in enumerate(contours):
            is_inner = False
            for outer_idx, outer_contour in enumerate(contours):
                if inner_idx == outer_idx: # ignore the same to compare itself
                    continue
                outer_rect = get_minimum_rect_from_contours([outer_contour])
                if is_contour_in_rect(inner_contour, outer_rect):
                    is_inner = True
                    break
            if (is_inner == True) :
                inner_contours.append(inner_contour)
        # Step 5.1.2 figure out center point of contour in inner_contour and sorting it by the order x
        print('z = {}, len(inner_contours) = {}'.format(z, len(inner_contours)))
        inner_cen_pts = []
        for contour in inner_contours:
            #rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            inner_cen_pts.append(cen_pt)
        inner_cen_pts.sort(key=lambda pt:pt[0])
        print('tandem first slice evaluation inner_cen_pts = {}'.format(inner_cen_pts))
        if(len(inner_cen_pts) != 3 ) :
            print('inner_cen_pts is not == 3')
            # To process first slice when there are no no three inner contour
            min_z = sorted(dicom_dict['z'].keys())[0]
            ct_obj = dicom_dict['z'][min_z]
            ps_x = ct_obj['ps_x']
            ps_y = ct_obj['ps_y']
            a_mm = 1.5 # allowed distance mm
            a_px_x = a_mm / ps_x # allowed distance pixel for x-axis
            a_px_y = a_mm / ps_y # allowed distance pixel for y-axis

            l_pt = lt_ovoid[0]
            r_pt = rt_ovoid[0]
            m_pt = None # middle pt
            for pt in inner_cen_pts:
                #if pt[0] > l_pt[0] + 4 and pt[0] < r_pt[0] - 4:
                if pt[0] > l_pt[0] + a_px_x and pt[0] < r_pt[0] - a_px_x:
                    #if (pt[1] > min([l_pt[1], r_pt[1]]) - 4 and pt[1] < max([l_pt[1], r_pt[1]]) + 4 ):
                    if (pt[1] > min([l_pt[1], r_pt[1]]) - a_px_y and pt[1] < max([l_pt[1], r_pt[1]]) + a_px_y):
                        m_pt = pt
                        break
            if (m_pt == None):
                # Use algo02 to find tandem
                # And remember to avoid needle_lines[][0]

                min_z = sorted(dicom_dict['z'].keys())[0]
                ct_obj = dicom_dict['z'][min_z]
                ps_x = ct_obj['ps_x']
                ps_y = ct_obj['ps_y']

                #ct_obj['output']['algo02']
                potential_contours = []
                for contour in ct_obj['output']['contours512']['algo02']:
                    rect_info = get_rect_info_from_cv_contour(contour)
                    cen_pt = (rect_info[2][0], rect_info[2][1])
                    inner_rect_gap_allowed_mm = 8.4
                    i_mm = inner_rect_gap_allowed_mm
                    i_px_x = i_mm / ps_x
                    i_px_y = i_mm / ps_y
                    #if cen_pt[0] > max(l_pt[0], r_pt[0])-28 or cen_pt[0]  < min(l_pt[0], r_pt[0])+28:
                    if cen_pt[0] > max(l_pt[0], r_pt[0]) - i_px_x or cen_pt[0] < min(l_pt[0], r_pt[0]) + i_px_x:
                        continue
                    #if cen_pt[1] > max(l_pt[1], r_pt[1])-28 or cen_pt[1]  < min(l_pt[1], r_pt[1])+28:
                    if cen_pt[1] > max(l_pt[1], r_pt[1]) - i_px_y or cen_pt[1] < min(l_pt[1], r_pt[1]) + i_px_y:
                        continue
                    cen_pt_is_on_needle = False
                    for needle_line in needle_lines:
                        n_pt = needle_line[0]
                        gap_of_needle_allowed_distance_mm = 1
                        dist_to_needle_mm = math.sqrt( ((n_pt[0]-cen_pt[0])*ps_x)**2 + ( (n_pt[1] - cen_pt[1])*ps_y)**2 )
                        if dist_to_needle_mm < gap_of_needle_allowed_distance_mm:
                            cen_pt_is_on_needle = True
                            break
                    if cen_pt_is_on_needle == True:
                        continue
                    potential_contours.append(contour)
                if len(potential_contours) == 1:
                    contour = potential_contours[0]
                    rect_info = get_rect_info_from_cv_contour(contour)
                    m_pt = (rect_info[2][0], rect_info[2][1])
                    tandem.append((m_pt[0], m_pt[1], float(z)))
                else:
                    if len(potential_contours) == 0 and len(needle_lines) == 1:
                        # In this case needle should be remove and change the needle to tandem
                        only_needle_line = needle_lines[0]
                        m_pt = only_needle_line[0]
                        needle_lines.remove(only_needle_line)
                        tandem.append( (m_pt[0], m_pt[1], float(z)) )
                        pass
                    else:
                        raise Exception

            else:
                tandem.append((m_pt[0], m_pt[1], float(z)))
        else :
            tandem.append( (inner_cen_pts[1][0], inner_cen_pts[1][1], float(z)) )

        # Step 5.1. Find Algo01 and detect the inner_contour

        print('TODO tandem for the case that without thicker pipe in scanned CT')

    # Step 6. Trace tandem
    print('Step 6. Trace tandem')
    # Step 6.1 Figure out [upper_half_z_idx_start, upper_half_z_idx_end) for upper-part of tandem
    z = sorted(dicom_dict['z'].keys())[0]
    last_z = tandem[-1][2]
    print('last_z = {}'.format(last_z))
    z_idx = sorted(dicom_dict['z'].keys()).index(last_z)
    print('z_idx of last_z={} is {}'.format(last_z, idx_z) )
    upper_half_z_idx_start = (z_idx + 1 )# upper_half_z_idx_start is the next z of last_z in current tandem data.
    print('upper_half_z_idx_start = {}'.format(upper_half_z_idx_start))
    upper_half_z_idx_end = len(dicom_dict['z'].keys())
    print('upper_half_z_idx_end = {}'.format(upper_half_z_idx_end))
    print('upper_half_z_idx [start,end) = [{},{}) '.format(upper_half_z_idx_start, upper_half_z_idx_end))
    z_start = sorted(dicom_dict['z'].keys())[upper_half_z_idx_start]
    z_end = sorted(dicom_dict['z'].keys())[upper_half_z_idx_end-1]+0.001
    print('and [start_z, end_z)  = [{},{})'.format(z_start,z_end))
    #print('upper_half_z_idx [start,end) = [{},{}) and z = [{},{})'.format(upper_half_z_idx_start, upper_half_z_idx_end, sorted(dicom_dict['z'].keys())[upper_half_z_idx_start],  sorted(dicom_dict['z'].keys())[upper_half_z_idx_end]  ))


    # Step 6.2 Setup first prev_info for loop to run and also set allowed_distnace to indicate the largest moving distance between two slice.
    # allowed_distance_mm = 8.5 # allowed distance when trace from bottom to tips of Tandem [ 8.5 mm is not ok for 35252020-2 ]
    #allowed_distance_mm = 10.95  # allowed distance when trace from bottom to tips of Tandem
    allowed_distance_mm = 10.95  # allowed distance when trace from bottom to tips of Tandem


    prev_info = {}
    prev_info['pt'] = tandem[-1]
    prev_info['ps_x'] = dicom_dict['z'][last_z]['ps_x']
    prev_info['ps_y'] = dicom_dict['z'][last_z]['ps_y']
    # The case for 29059811-2 folder , will have the following value
    # last_z == -92.0
    # prev_info == {'pt': (240, 226, -92.0), 'ps_x': "3.90625e-1", 'ps_y': "3.90625e-1"}

    # Step 6.3. Start to trace tandem
    for z_idx in range(upper_half_z_idx_start, upper_half_z_idx_end):
        z = sorted(dicom_dict['z'].keys())[z_idx]
        if (z == -34):
            print('z is -34 arrive')
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']

        #print('z = {}'.format(z))
        # Step 6.3.1. Make contours variable as collecting of all contour in z-slice
        contours = []
        for algo_key in dicom_dict['z'][z]['output']['contours512'].keys():
            contours = contours + dicom_dict['z'][z]['output']['contours512'][algo_key]
        # Step 6.3.2. Convert to center pt that the idx is the same as to contours
        cen_pts = []
        for c_idx, c in enumerate(contours):
            rect_info = get_rect_info_from_cv_contour(c)
            # [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            cen_pt = (rect_info[2][0], rect_info[2][1])
            cen_pts.append(cen_pt)
        # Step 6.3.3. Find closed center point for these center pt. And append it into tandem line. But leave looping if there is no center pt
        # prev_info is like {'pt': (240, 226, -92.0), 'ps_x': "3.90625e-1", 'ps_y': "3.90625e-1"}
        # pt in cen_pts is like (240, 226)
        minimum_distance_mm = allowed_distance_mm + 1  # If minimum_distance_mm is finally large than allowed_distance_mm, it's mean there is no pt closed to prev_pt
        minimum_pt = (0, 0)
        print('cen_pts = {}'.format(cen_pts ))
        for pt in cen_pts:
            prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
            prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
            x_mm = pt[0] * ps_x
            y_mm = pt[1] * ps_y
            distance_mm = math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2 )
            if (distance_mm > allowed_distance_mm ) : # distance_mm cannot large than allowed_distance_mm
                continue
            if (distance_mm < minimum_distance_mm):
                minimum_distance_mm = distance_mm
                minimum_pt = pt

        if (minimum_distance_mm > allowed_distance_mm):
            # This is case to say ending for the upper looper
            print('more than tip ')
            break
        else:
            tandem.append( (minimum_pt[0], minimum_pt[1],float(z)) )
            prev_info['pt'] = minimum_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            print('tandem = {}'.format(tandem))
    return (lt_ovoid, tandem, rt_ovoid)
def algo_to_get_needle_lines(dicom_dict):
    """
    After horizontal algorithm have figured out by Horizontal Algorithm,
    algo_to_get_needle_lines() ia processing in vertical for every ct slice in dicom_dict.
    This algorithm will figure out array of needle line and each line is with unit of pixel

    :param dicom_dict:
    :return:
        return array of needle lines
    """
    from utilities import get_rect_info_from_cv_contour
    needle_lines = []
    # Step 1. Use algo07 to get center point of inner contour
    last_z_in_step1 = sorted(dicom_dict['z'].keys())[0]
    center_pts_dict = {}
    for z in sorted(dicom_dict['z'].keys()):
        contours = dicom_dict['z'][z]['output']['contours512']['algo07']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo07')
        center_pts_dict[z] = []
        for contour in contours:
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            center_pts_dict[z].append(cen_pt)
        center_pts_dict[z].sort(key=lambda pt:pt[0])
        print('center_pts_dict[{}] = {}'.format(z, center_pts_dict[z]))

    print('STEP 2.')

    min_z = sorted(center_pts_dict.keys())[0]
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of L't Ovoid
    # Get first slice and see how many needle point in it. the index of needle point in first slice will be the needle_line_idx
    for needle_line_idx in range(len(center_pts_dict[min_z])):
        print('needle_line_idx = {}'.format(needle_line_idx))
        needle_line = []
        prev_info = {}
        prev_info['pt'] = None
        prev_info['ps_x'] = None
        prev_info['ps_y'] = None
        for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
            ps_x = dicom_dict['z'][z]['ps_x']
            ps_y = dicom_dict['z'][z]['ps_y']
            if idx_z == 0:
                #prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
                prev_pt = (center_pts_dict[z][needle_line_idx][0], center_pts_dict[z][needle_line_idx][1], float(z))
                prev_info['pt'] = prev_pt
                prev_info['ps_x'] = ps_x
                prev_info['ps_y'] = ps_y
                #lt_ovoid.append(prev_pt)
                needle_line.append(prev_pt)
                continue
            prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
            prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
            #x_mm = center_pts_dict[z][0][0] * ps_x
            if( needle_line_idx >= len(center_pts_dict[z]) ):
                # It's mean there is no more point, so continue
                continue
            #argmin_idx = 0
            potential_list = []
            for pt_idx, pt in enumerate(center_pts_dict[z]):
                x_mm = center_pts_dict[z][pt_idx][0] * ps_x
                y_mm = center_pts_dict[z][pt_idx][1] * ps_y
                dist_mm = math.sqrt((x_mm - prev_x_mm) ** 2 + (y_mm - prev_y_mm) ** 2)
                if dist_mm < allowed_distance_mm:
                    potential_list.append( (pt_idx , dist_mm) )
            sorted_potential_list = sorted(potential_list, key=lambda item:item[1])
            if len(sorted_potential_list) <= 0 :
                print('There is not closed point')
                break
            pt_idx = sorted_potential_list[0][0]
            prev_pt = (center_pts_dict[z][pt_idx][0], center_pts_dict[z][pt_idx][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            #lt_ovoid.append(prev_pt)
            needle_line.append(prev_pt)
            print('needle_line (with idx={})  = {}'.format(needle_line_idx, needle_line))



            """
            x_mm = center_pts_dict[z][needle_line_idx][0] * ps_x #Error
            #y_mm = center_pts_dict[z][0][1] * ps_y
            y_mm = center_pts_dict[z][needle_line_idx][1] * ps_y
            if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
                #prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
                prev_pt = (center_pts_dict[z][needle_line_idx][0], center_pts_dict[z][needle_line_idx][1], float(z))
                prev_info['pt'] = prev_pt
                prev_info['ps_x'] = ps_x
                prev_info['ps_y'] = ps_y
                #lt_ovoid.append(prev_pt)
                needle_line.append(prev_pt)
                print('needle_line (with idx={})  = {}'.format(needle_line_idx, needle_line))
            else:
                print('allowed_distance_mm = {}'.format(allowed_distance_mm))
                print('math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) = {}'.format(math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2)))
                break
            """
        needle_lines.append(needle_line)
    return needle_lines

# FUNCTIONS - main genearte function
def generate_brachy_rp_file(RP_OperatorsName, dicom_dict, out_rp_filepath, is_enable_print=False):
    """

    :param RP_OperatorsName:
        The value will assign to Operators Name field in RP file
    :param dicom_dict:
        Every folder will incldue RP, RD, RS and CTs files. The dicom_dict is dictionary the represent all these information that we need in this folder
    :param out_rp_filepath:
        The RP file will save to out_rp_filepath
    :param is_enable_print:
        It will show all of printing log if is_eanble_print == True
        It will not any printing log if is_enable_print == False
        By the way, when you set is_enable_print as False but you just need to print some information in some line,
        All you cna do is use enablePrint() and blockPrint() function. For example,
            ...
            enablePrint()
            print('This is for specific debug when you have some confuse on some code and you want to open is_enable_print')
            blockPrint()
            ...
    :return:
        No return value
    """
    from utilities import wrap_to_rp_file
    from utilities import get_metric_lines_representation
    from utilities import get_metric_needle_lines_representation
    from utilities import get_applicator_rp_line
    from utilities import get_HR_CTV_min_z
    is_enable_print=True
    if (is_enable_print == False):
        blockPrint()
    else:
        enablePrint()
    # Step 1. Get line of lt_ovoid, tandem, rt_ovoid by OpneCV contour material and innovated combination
    needle_lines = algo_to_get_needle_lines(dicom_dict)
    print('len(needle_lines) = {}'.format(len(needle_lines)))
    if len(needle_lines) > 0 :
        for idx, needle_line in enumerate(needle_lines):
            print('needle_lines[{}] = {}'.format(idx, needle_lines[idx]))

    (lt_ovoid, tandem, rt_ovoid) = algo_to_get_pixel_lines(dicom_dict, needle_lines)

    # Step 2. Convert line into metric representation
    # Original line is array of (x_px, y_px, z_mm) and we want to convert to (x_mm, y_mm, z_mm)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
    print('metric_lt_ovoid = {}'.format(metric_lt_ovoid))
    print('metric_tandem = {}'.format(metric_tandem))
    print('metric_rt_ovoid = {}'.format(metric_rt_ovoid))


    metric_needle_lines = get_metric_needle_lines_representation(dicom_dict, needle_lines)

    # Step 2.1 Extend metric needle line with 2mm in the end point of metric_needle_lines
    needle_extend_mm = 2
    for line_idx, line in enumerate(metric_needle_lines):
        pt_s = line[0] # point start
        pt_e = line[-1] # point end
        pt_n = pt_e.copy() # point new. it will append in end of line
        cur_dist = math.sqrt( (pt_e[0]-pt_s[0])**2 + (pt_e[1]-pt_s[1])**2 + (pt_e[2]-pt_s[2])**2 )
        for i in range(3):
            pt_n[i] = pt_n[i] + ( (pt_e[i] - pt_s[i]) * (needle_extend_mm / cur_dist) )
        line.append(pt_n)

    print('len(metric_needle_lines) = {}'.format(len(metric_needle_lines)))
    for line_idx, line in enumerate(metric_needle_lines):
        print('metric_needle_lines[{}]= {}'.format(line_idx, line))

    # Step 3. Reverse Order, so that first element is TIPS [from most top (z maximum) to most bottom (z minimum) ]
    metric_lt_ovoid.reverse()
    metric_tandem.reverse()
    metric_rt_ovoid.reverse()
    for metric_line in metric_needle_lines:
        metric_line.reverse()

    # Step 4. Get Applicator RP line
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 4, 5)

    # for debug , so change about testing rp import correct or not. So change tandem start from 3mm to 13mm
    tandem_rp_line = get_applicator_rp_line(metric_tandem, 3, 5) # <-- change to reduce 1mm
    lt_ovoid_rp_line = get_applicator_rp_line(metric_lt_ovoid, 0, 5)
    rt_ovoid_rp_line = get_applicator_rp_line(metric_rt_ovoid, 0 ,5)
    rp_needle_lines = []
    for metric_line in metric_needle_lines:
        rp_needle_line = get_applicator_rp_line(metric_line, 0, 5)
        rp_needle_lines.append(rp_needle_line)

    print('lt_ovoid_rp_line = {}'.format(lt_ovoid_rp_line))
    print('tandem_rp_line = {}'.format(tandem_rp_line))
    print('rt_ovoid_rp_line = {}'.format(rt_ovoid_rp_line))
    print('len(rp_needle_lines) = {}'.format(len(rp_needle_lines)))
    for line_idx, line in enumerate(rp_needle_lines):
        print('rp_needle_lines[{}]= {}'.format(line_idx, line))


    # Step 4.2 Delete the point in the rp lines that z < z_target
    z_target = get_HR_CTV_min_z(dicom_dict['pathinfo']['rs_filepath']) - 20
    print(tandem_rp_line)
    tandem_rp_line = [pt for pt in tandem_rp_line if (pt[2] > z_target)]
    print(tandem_rp_line)
    lt_ovoid_rp_line = [pt for pt in lt_ovoid_rp_line if (pt[2] > z_target)]
    rt_ovoid_rp_line = [pt for pt in rt_ovoid_rp_line if (pt[2] > z_target)]
    for r_idx in range(len(rp_needle_lines)):
        rp_needle_lines[r_idx] = copy.deepcopy([pt for pt in rp_needle_lines[r_idx] if (pt[2] > z_target)])


    # Step 5. Wrap to RP file
    rs_filepath = dicom_dict['pathinfo']['rs_filepath']
    print('out_rp_filepath = {}'.format(out_rp_filepath))
    applicator_roi_dict = dicom_dict['metadata']['applicator_roi_dict']
    wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,
                    out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=rp_needle_lines,
                    rt_ovoid_rp_line=rt_ovoid_rp_line, applicator_roi_dict=applicator_roi_dict)
    if (is_enable_print == False):
        enablePrint()
def generate_brachy_rp_file_without_needle(RP_OperatorsName, dicom_dict, out_rp_filepath, is_enable_print=False):
    """
    [Generate RP file bh inputs and we don't figure needle in this function]
    :param RP_OperatorsName:
        The value will assign to Operators Name field in RP file
    :param dicom_dict:
        Every folder will incldue RP, RD, RS and CTs files. The dicom_dict is dictionary the represent all these information that we need in this folder
    :param out_rp_filepath:
        The RP file will save to out_rp_filepath
    :param is_enable_print:
        It will show all of printing log if is_eanble_print == True
        It will not any printing log if is_enable_print == False
        By the way, when you set is_enable_print as False but you just need to print some information in some line,
        All you cna do is use enablePrint() and blockPrint() function. For example,
            ...
            enablePrint()
            print('This is for specific debug when you have some confuse on some code and you want to open is_enable_print')
            blockPrint()
            ...
    :return:
        No return value
    """
    from utilities import wrap_to_rp_file
    from utilities import get_metric_lines_representation
    from utilities import get_metric_needle_lines_representation
    from utilities import get_applicator_rp_line
    from utilities import get_HR_CTV_min_z

    if (is_enable_print == False):
        blockPrint()
    else:
        enablePrint()
    print('Call generate_brachy_rp_file_without_needle()')

    # Step 1. Get line of lt_ovoid, tandem, rt_ovoid by OpneCV contour material and innovated combination
    (lt_ovoid, tandem, rt_ovoid) = algo_to_get_pixel_lines(dicom_dict)
    # Step 2. Convert line into metric representation
    # Original line is array of (x_px, y_px, z_mm) and we want to convert to (x_mm, y_mm, z_mm)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
    print('metric_lt_ovoid = {}'.format(metric_lt_ovoid))
    print('metric_tandem = {}'.format(metric_tandem))
    print('metric_rt_ovoid = {}'.format(metric_rt_ovoid))

    # Step 3. Reverse Order, so that first element is TIPS [from most top (z maximum) to most bottom (z minimum) ]
    metric_lt_ovoid.reverse()
    metric_tandem.reverse()
    metric_rt_ovoid.reverse()

    # Step 4. Get Applicator RP line
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 4, 5)

    # for debug , so change about testing rp import correct or not. So change tandem start from 3mm to 13mm
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 3, 5) # <-- change to reduce 1mm
    tandem_rp_line = get_applicator_rp_line(metric_tandem, 13, 5)  # <-- change to reduce 1mm
    lt_ovoid_rp_line = get_applicator_rp_line(metric_lt_ovoid, 0, 5)
    rt_ovoid_rp_line = get_applicator_rp_line(metric_rt_ovoid, 0 ,5)
    print('lt_ovoid_rp_line = {}'.format(lt_ovoid_rp_line))
    print('tandem_rp_line = {}'.format(tandem_rp_line))
    print('rt_ovoid_rp_line = {}'.format(rt_ovoid_rp_line))

    # Step 5. Wrap to RP file
    # TODO for wrap rp_needle_lines into RP file
    print(dicom_dict['pathinfo']['rs_filepath'])
    print(dicom_dict['metadata'].keys())
    rs_filepath = dicom_dict['pathinfo']['rs_filepath']

    print('out_rp_filepath = {}'.format(out_rp_filepath))
    applicator_roi_dict = dicom_dict['metadata']['applicator_roi_dict']
    wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,
                    out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=[],
                    rt_ovoid_rp_line=rt_ovoid_rp_line, applicator_roi_dict=applicator_roi_dict)

    if (is_enable_print == False):
        enablePrint()

# FUNCTIONS - Some ploting utility functions support for you to check CT pictures with data
def generate_all_rp_process(
        root_folder=r'RAL_plan_new_20190905', rp_output_folder_filepath='all_rp_output',  bytes_dump_folder_filepath='contours_bytes',
        is_recreate_bytes=True, debug_folders=[]):

    """

    :param root_folder:
        The folder structure of current data is like {root_folder}/{case_folder}
        people always put many case folder in root_folder.
        You can generate each dicom_dict for each case_folder. And then process this case into RP file

    :param rp_output_folder_filepath:
        Every RP file will put into

    :param bytes_dump_folder_filepath:
    :param is_recreate_bytes:
    :param debug_folders:
    :return:
    """
    from utilities import generate_metadata_to_dicom_dict
    from utilities import create_directory_if_not_exists
    from utilities import python_object_dump
    from utilities import python_object_load

    print('Call generate_all_rp_process with the following arguments')
    print('root_folder = ', root_folder)
    print('rp_output_folder_filepath = ', rp_output_folder_filepath)
    print('bytes_dump_folder_filepath = ', bytes_dump_folder_filepath)

    create_directory_if_not_exists(bytes_dump_folder_filepath)
    create_directory_if_not_exists(rp_output_folder_filepath)

    print('[START] generate_all_rp_process()')
    all_dicom_dict = {}
    # Step 2. Generate all our target
    f_list = [ os.path.join(root_folder, file) for file in os.listdir(root_folder) ]
    folders = os.listdir(root_folder)
    total_folders = []
    failed_folders = []
    success_folders = []
    sorted_f_list = copy.deepcopy(sorted(f_list))

    for folder_idx, folder in enumerate(sorted_f_list):
        enablePrint()
        if len(debug_folders) != 0:
            if (os.path.basename(folder) not in debug_folders):
                continue

        print('\n[{}/{}] Loop info : folder_idx = {}, folder = {}'.format(folder_idx + 1, len(folders), folder_idx, folder),flush=True)
        byte_filename = r'{}.bytes'.format(os.path.basename(folder))
        dump_filepath = os.path.join(bytes_dump_folder_filepath, byte_filename)

        if (is_recreate_bytes == True):
            time_start = datetime.datetime.now()
            print('[{}/{}] Create bytes file {} '.format(folder_idx + 1, len(folders), dump_filepath), end=' -> ',flush=True)
            dicom_dict = get_dicom_dict(folder)
            generate_metadata_to_dicom_dict(dicom_dict)
            generate_output_to_dicom_dict(dicom_dict)
            all_dicom_dict[folder] = dicom_dict
            python_object_dump(dicom_dict, dump_filepath)
            time_end = datetime.datetime.now()
            print('{}s [{}-{}]'.format(time_end - time_start, time_start, time_end), end='\n', flush=True)
        else: # CASE is_recreate_bytes == False
            bytes_filepath = os.path.join(bytes_dump_folder_filepath, r'{}.bytes'.format(os.path.basename(folder)))
            bytes_file_exists = os.path.exists(bytes_filepath)
            if bytes_file_exists == True:
                #dicom_dict = python_object_load(bytes_filepath)
                #all_dicom_dict[folder] = dicom_dict
                print('[{}/{}] File have been created - {}'.format(folder_idx + 1, len(folders), dump_filepath), flush=True)
            else: #CASE When the file is not exist in bytes_filepath
                time_start = datetime.datetime.now()
                print('[{}/{}] Create bytes file {} '.format(folder_idx + 1, len(folders), dump_filepath), end=' -> ',flush=True)
                dicom_dict = get_dicom_dict(folder)
                generate_metadata_to_dicom_dict(dicom_dict)
                generate_output_to_dicom_dict(dicom_dict)
                all_dicom_dict[folder] = dicom_dict
                python_object_dump(dicom_dict, dump_filepath)
                time_end = datetime.datetime.now()
                print('{}s [{}-{}]'.format(time_end - time_start, time_start, time_end), end='\n', flush=True)
        # Change to basename of folder here
        fullpath_folder = folder
        folder = os.path.basename(folder)
        total_folders.append(folder)
        try:
            #bytes_filepath = os.path.join('contours_bytes', r'{}.bytes'.format(folder))
            bytes_filepath = os.path.join(bytes_dump_folder_filepath, r'{}.bytes'.format(folder))
            dicom_dict = python_object_load(bytes_filepath)
            if fullpath_folder not in all_dicom_dict.keys():
                all_dicom_dict[fullpath_folder] = dicom_dict

            metadata = dicom_dict['metadata']
            # out_rp_filepath format is PatientID, RS StudyDate  and the final is folder name processing by coding

            out_rp_filepath = r'RP.{}.{}.f{}.dcm'.format(  metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder']) )
            #out_rp_filepath = os.path.join('all_rp_output', out_rp_filepath)
            out_rp_filepath = os.path.join(rp_output_folder_filepath, out_rp_filepath)
            time_start = datetime.datetime.now()
            print('[{}/{}] Create RP file -> {}'.format(folder_idx+1,len(folders), out_rp_filepath) ,end=' -> ', flush=True)
            #generate_brachy_rp_file_without_needle(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath,is_enable_print=False)
            generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=False)
            #generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=True)
            time_end = datetime.datetime.now()
            print('{}s [{}-{}]'.format(time_end-time_start, time_start, time_end), end='\n', flush=True)
            success_folders.append(folder)

            # Added code for output RS file and output RP file into folder by each patient'study case
            import_root_folder = r'import_output'
            import_folder_filepath = os.path.join(import_root_folder, r'{}.{}.f{}'.format(metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder'])))
            create_directory_if_not_exists(import_folder_filepath)
            src_rp_filepath = out_rp_filepath
            dst_rp_filepath = os.path.join(import_folder_filepath, os.path.basename(out_rp_filepath))
            print('src_rp_filepath = {}'.format(src_rp_filepath))
            print('dst_rp_filepath = {}'.format(dst_rp_filepath))
            copyfile(src_rp_filepath, dst_rp_filepath)
            src_rs_filepath = dicom_dict['pathinfo']['rs_filepath']
            dst_rs_filepath = os.path.join(import_folder_filepath, os.path.basename(src_rs_filepath))
            print('src_rs_filepath = {}'.format(src_rs_filepath))
            print('dst_rs_filepath = {}'.format(dst_rs_filepath))
            copyfile(src_rs_filepath, dst_rs_filepath)
        except Exception as debug_ex:
            print('Create RP file Failed')
            failed_folders.append(folder)
            print(debug_ex)
            #raise debug_ex
    print('FOLDER SUMMARY REPORT')
    print('failed folders = {}'.format(failed_folders))
    print('failed / total = {}/{}'.format(len(failed_folders), len(total_folders) ))
    print('success /total = {}/{}'.format(len(success_folders), len(total_folders) ))

    # Step 3. Use python_object_dump to dump it into some file
    try:
        print('Creating {} in very largest size'.format(filename))
        python_object_dump(all_dicom_dict, filename)
        print('Created {}'.format(filename))
    except Exception as ex:
        print('Create largest size dicom file failed')
    print('[END] generate_all_rp_process()')
def main():
    # 10 CASE
    # print('root_folder = Study-RAL-implant_20191112 -> {}'.format(
    #     [os.path.basename(item) for item in os.listdir('Study-RAL-implant_20191112')]))
    # generate_all_rp_process(root_folder=r'Study-RAL-implant_20191112',
    #                         rp_output_folder_filepath='Study-RAL-implant_20191112_RP_Files',
    #                         bytes_dump_folder_filepath='Study-RAL-implant_20191112_Bytes_Files',
    #                         is_recreate_bytes=True, debug_folders=[])
    # '804045'

    # 31 CASE
    # print('root_folder = RAL_plan_new_20190905 -> {}'.format(
    #     [os.path.basename(item) for item in os.listdir('RAL_plan_new_20190905')]))
    # generate_all_rp_process(root_folder=r'RAL_plan_new_20190905',
    #                         rp_output_folder_filepath='RAL_plan_new_20190905_RP_Files',
    #                         bytes_dump_folder_filepath='RAL_plan_new_20190905_Bytes_Files',
    #                         is_recreate_bytes=True, debug_folders=[])

    # 22 CASE : the case of 33220132 is only one tandem and not with pipe. This case should be wrong
    print('root_folder = Study-RAL-20191105 -> {}'.format(
        [os.path.basename(item) for item in os.listdir('Study-RAL-20191105')]))
    generate_all_rp_process(root_folder=r'Study-RAL-20191105',
                            rp_output_folder_filepath='Study-RAL-20191105_RP_Files',
                            bytes_dump_folder_filepath='Study-RAL-20191105_Bytes_Files',
                            is_recreate_bytes=True, debug_folders=[])

def generate_rp_by_ct_rs_folder(input_ct_rs_folder, output_rp_filepath):
    from utilities import generate_metadata_to_dicom_dict
    folder = input_ct_rs_folder
    dicom_dict = get_dicom_dict(folder)
    generate_metadata_to_dicom_dict(dicom_dict)
    generate_output_to_dicom_dict(dicom_dict)
    generate_brachy_rp_file(RP_OperatorsName='cylin',
                            dicom_dict=dicom_dict,
                            out_rp_filepath=output_rp_filepath,
                            is_enable_print=False)
if __name__ == '__main__':
    #main()
    generate_rp_by_ct_rs_folder(
        input_ct_rs_folder = r"RAL_plan_new_20190905\29059811-1",
        output_rp_filepath = r"RP.output.dcm")








