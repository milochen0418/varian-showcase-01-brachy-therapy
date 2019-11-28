import os
import sys
import numpy as np
import pydicom
import copy


CT_table = {}
Thickness_table = {}
lastGetDateFromCT = None
SOPInstanceUID_list_of_CTs = None



def pydicom_read_rs_template(ct_filelist, rs_template_filepath):
    print("get_temp_rs() is calling with rs_template_filepath = ", rs_template_filepath, ", and ct_filelist = ", ct_filelist)
    titles = {'Accession Number',
              'Ethnic Group',
              'Instance Creation Date',
              'Patient ID',
              "Patient's Birth Date",
              "Patient's Name",
              "Patient's Sex",
              "Referring Physician's Name",
              'Specific Character Set',
              'Study Date',
              'Study ID',
              'Study Instance UID',
              'Study Time'}
    ct_filepath = ct_filelist[0]
    ct_fp = pydicom.dcmread(ct_filepath)
    ct_fp_title = dict()
    for key in ct_fp.keys():
        ct_fp_title[ ct_fp[key].name ] = key

    # trs = pydicom.dcmread(r"D:\WatchFolder\res\RS.1.2.246.352.71.4.650410132.39376.20101029115033.dcm")
    rs_fp = pydicom.dcmread(rs_template_filepath)
    rs_fp_title = dict()
    for key in rs_fp.keys():
        rs_fp_title[rs_fp[key].name] = key

    for title in titles:
        if title in ct_fp_title:
            rs_fp[rs_fp_title[title]].value = ct_fp[ct_fp_title[title]].value
    return rs_fp


def make_SOPInstnaceUID_list_of_CTs(filelist):
    global SOPInstanceUID_list_of_CTs
    #ct_folder_filepath = Env.REMOTE_FOLDER_OF_CT_WITH_PID
    SOPInstanceUID_list_of_CTs = []
    #ct_folder = ct_folder_filepath
    #for file in os.listdir(ct_folder):
    for filepath in filelist:
        #filepath = "{}\\{}".format(ct_folder, file)
        ct_fp = None
        try:
            ct_fp = pydicom.read_file(filepath)
        except:
            pass
        if ct_fp == None:
            continue

        #print( "z={} -> uid={}".format(ct_fp.ImagePositionPatient[2], ct_fp.SOPInstanceUID) )
        uid = ct_fp.SOPInstanceUID
        SOPInstanceUID_list_of_CTs.append(uid)

def make_some_tables(input_ct_filelist):
    print("make_some_tables")
    global CT_table
    global Thickness_table
    global lastGetDateFromCT
    global SOPInstanceUID_list_of_CTs
    #folder_path = Env.FOLDER_PATH_OF_CT
    #folder_path = Env.REMOTE_FOLDER_OF_CT_WITH_PID
    #print("make_some_table() -> folder_path = ", folder_path)

    for ctdcm in input_ct_filelist:
        filepath = ctdcm
        #print("filepath = ", filepath)
        ct_fp = None
        try :
            ct_fp = pydicom.read_file(filepath)
        except:
            pass
        if ct_fp == None:
            continue
        #print("passed filepath = ", filepath)
        lastGetDateFromCT = ct_fp.StudyDate

        z = float(ct_fp.ImagePositionPatient[2])
        uid = ct_fp.SOPInstanceUID
        thickness = ct_fp.SliceThickness
        CT_table[z] = uid
        Thickness_table[z] = thickness
        #print("z = ", z, ", uid = ",uid, ", Thickness_table = ", Thickness_table)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def interpolate_and_wrapup_rs(input_mrcnn_out, input_ct_filelist, output_rs_filepath_after_ai):
    #RS_TEMPLATE_FILEPATH = r"C:/Users/Milo/Desktop/Milo/ModelsAndRSTemplates/Brachy/RS_Template/RS.1.2.246.352.71.4.417454940236.267194.20190411111011.dcm"
    RS_TEMPLATE_FILEPATH = r"../ModelsAndRSTemplates/Brachy/RS_Template/RS.1.2.246.352.71.4.417454940236.267194.20190411111011.dcm"
    print("call interpolate_and_wrapup_rs()")
    #print("with arg input_mrcnn_out  = {}", input_mrcnn_out)
    print("with arg input_ct_filelist = {}", input_ct_filelist)
    print("with arg output_rs_filepath_after_ai = {}", output_rs_filepath_after_ai)
    import datetime
    import traceback
    from pydicom.dataset import Dataset, DataElement
    #from pydicom.dataset import Dataset, DataElement

    #save rs file into the filepath = output_rs_filepath_after_ai
    global CT_table
    global Thickness_table
    CT_table = {}
    Thickness_table = {}


    rs_fp = pydicom_read_rs_template(input_ct_filelist, RS_TEMPLATE_FILEPATH)

    ct_FrameOfReferenceUID = None
    ct_StudyInstnaceUID = None
    for filepath in input_ct_filelist:
        ct_fp = pydicom.read_file(filepath)
        ct_FrameOfReferenceUID = ct_fp.FrameOfReferenceUID
        ct_StudyInstanceUID = ct_fp.StudyInstanceUID
        break


    #now = datetime.datetime.now()
    #theDateForSOPInstanceUID = now.strftime("%Y%m%d%H%M%S")
    #uid = "1.2.246.352.71.4.417454940236.267194.{}".format(theDateForSOPInstanceUID)




    # ================== process_rs() start ===========================
    global SOPInstanceUID_list_of_CTs



    label_id_mask = input_mrcnn_out
    labels = list(label_id_mask.keys())




    # =================== Generate Color Mapping ===========================
    print("print(labels)=============================================")
    print(labels)
    # ['bowel', 'Uterus', 'HR-CTV', 'Sigmoid_colon', 'Rectum', 'Bladder', 'Foley']
    print("print(labels)=============================================")

    # For Brachy template, labels is like ['bowel', 'Uterus', 'HR-CTV', 'Sigmoid_colon', 'Rectum', 'Bladder', 'Foley']
    colors = []
    # disable code by milochen
    #rs_fp = pydicom.read_file(Env.RS_TEMPLATE_FILEPATH)

    # Make labels->colors mapping table by information of original template ADD by milochen
    # The output result should look like
    # ['bowel', 'Uterus', 'HR-CTV', 'Sigmoid_colon', 'Rectum', 'Bladder', 'Foley']
    #
    # [None, ['255', '255', '128'], ['0', '255', '0'], ['187', '255', '187'], ['128', '64', '64'], ['0', '255', '255'],['0', '150', '0'] ]
    for lbl_idx in range(len(labels)):
        lblName = labels[lbl_idx]

        # Find the ROI Number that its ROIName equal to label name lblName
        isFindMatchedROINumber = False
        matchedROINumber = None
        for idx in range(len(rs_fp.StructureSetROISequence)):
            item = rs_fp.StructureSetROISequence[idx]
            if item.ROIName == lblName:
                isFindMatchedROINumber = True
                matchedROINumber = item.ROINumber
                break
        if isFindMatchedROINumber == False:
            # not Matched , so ignore this stable
            print("isFindMatchedROINumber is False when labels[", lbl_idx, "] = ", lblName)
            colors.append(None)
            continue

        # Find the ROI Display Color that its matchedROINumber equal to ReferencedROINumber
        isFindMatchedROIDisplayColor = False
        matchedROIDisplayColor = None
        for idx in range(len(rs_fp.ROIContourSequence)):
            item = rs_fp.ROIContourSequence[idx]
            if item.ReferencedROINumber == matchedROINumber:
                isFindMatchedROIDisplayColor = True
                matchedROIDisplayColor = item.ROIDisplayColor
                break

        if isFindMatchedROIDisplayColor == False:
            print("isFindMatchedROIDisplayColor is False when labels[", lbl_idx, "] = ", lblName)
            colors.append(None)
            continue
        colors.append(matchedROIDisplayColor)

    # Make color mapping
    colorMapping = {}
    for idx in range(len(labels)):
        colorMapping[ labels[idx] ] = colors[idx]
        print(labels[idx], "->", colors[idx])

    print("colorMapping Research")
    print(labels)
    print(colorMapping)
    # The output table may look like this
    # bowel -> None
    # Uterus -> ['255', '255', '128']
    # HR - CTV -> ['0', '255', '0']
    # Sigmoid_colon -> ['187', '255', '187']
    # Rectum -> ['128', '64', '64']
    # Bladder -> ['0', '255', '255']
    # Foley -> ['0', '150', '0']
    #  check key for "if key in colorMapping"
    # Hightlight problem of code. ROINumber are change, so the order in original rs file is different to output rs file
    # The strategy of Sac's original code is to clean all of things and rewrite all.

    # =================== Generate Color Mapping ===========================




    ct_id = set()
    for ct_ids in label_id_mask.values():
        ct_id.update(ct_ids.keys())
    ct_id = list(ct_id)
    ct_id.sort()


    # fix bug of SeriesInstnaceUID, it should be a unique SeriesInstnaceUID
    # 1.2.246.352.71.2.417454940236.3986270.2019041110054111
    import datetime
    now = datetime.datetime.now()
    theDateForSeriesInstanceUID = now.strftime("%Y%m%d%H%M%S%f")
    rs_fp.SeriesInstanceUID = "1.2.246.352.71.2.417454940236.3986270." + theDateForSeriesInstanceUID
    # The format of our new SeriesInstanceUID is like this 1.2.246.352.71.2.417454940236.3986270.20190627171748535000

    rs_fp.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID = ct_FrameOfReferenceUID
    rs_fp.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = rs_fp.StudyInstanceUID
    rs_fp.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.clear()

    print("print_ct_id for check RTReferencedSeriesSequence ContourImageSequence")
    print(ct_id)
    global SOPInstanceUID_list_of_CTs
    make_SOPInstnaceUID_list_of_CTs(input_ct_filelist)
    # Query here https://imagej.nih.gov/nih-image/download/nih-image_spin-offs/NucMed_Image/DICOM%20Dictionary
    #for _id in ct_id:
    for _id in SOPInstanceUID_list_of_CTs:
        #print("_id = ", _id)
        ds = Dataset()
        ds[0x0008, 0x1150] = DataElement(0x00081150, 'UI', 'CT Image Storage') # Referenced SOP Class UID
        ds.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds[0x0008, 0x1155] = DataElement(0x00081155, 'UI', _id) # Referenced SOP Instance UID
        rs_fp.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(ds)

    rs_fp.StructureSetROISequence.clear()



    for i in range(len(labels)):
        ds = Dataset()
        ds[0x3006, 0x0022] = DataElement(0x30060022, 'IS', str(i + 1)) # ROI Number
        ds[0x3006, 0x0024] = DataElement(0x30060024, 'UI', ct_FrameOfReferenceUID) # Referenced Frame of Reference UID
        ds[0x3006, 0x0026] = DataElement(0x30060026, 'LO', labels[i]) # ROI Name
        ds[0x3006, 0x0036] = DataElement(0x30060036, 'CS', 'MANUAL') # ROI Generation Algorithm
        rs_fp.StructureSetROISequence.append(ds)


    rs_fp.ROIContourSequence.clear()

    for i in range(len(labels)):
        dsss = Dataset()
        # changed by milochen
        # dsss[0x3006, 0x002a] = DataElement(0x3006002a, 'IS', color)
        # Draw color
        lblName = labels[i]
        if lblName in colorMapping:
            drawColor = colorMapping[lblName]
            if drawColor == None:
                dsss[0x3006, 0x002a] = DataElement(0x3006002a, 'IS', [0, 0, 0]) # ROI Display Color
            else:
                dsss[0x3006, 0x002a] = DataElement(0x3006002a, 'IS', drawColor) # ROI Display Color
        else :
            dsss[0x3006, 0x002a] = DataElement(0x3006002a, 'IS', [255, 255, 255]) # ROI Display Color

        dsss.ContourSequence = []
        dsss[0x3006, 0x0084] = DataElement(0x30060084, 'IS', str(i + 1)) # Referenced ROI Number
        rs_fp.ROIContourSequence.append(dsss)

        for _id in label_id_mask[labels[i]].keys():
            ds = Dataset()
            ds[0x0008, 0x1150] = DataElement(0x00081150, 'UI', 'CT Image Storage') # SOP Class UID
            ds.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ds[0x0008, 0x1155] = DataElement(0x00081155, 'UI', _id) # SOP Instance UID

            NumberOfContourPoints = str(int(len(label_id_mask[labels[i]][_id]) / 3))

            dss = Dataset()
            dss.ContourImageSequence = [ds]
            dss[0x3006, 0x0042] = DataElement(0x30060042, 'CS', 'CLOSED_PLANAR') # Contour Geometric Type
            # dss[0x3006, 0x0046] = DataElement(0x30060046, 'IS', '664')
            dss[0x3006, 0x0046] = DataElement(0x30060046, 'IS', NumberOfContourPoints) # Number Of Contour Points
            dss[0x3006, 0x0050] = DataElement(0x30060050, 'DS', label_id_mask[labels[i]][_id]) # Contour Data
            rs_fp.ROIContourSequence[i].ContourSequence.append(dss)

    rs_fp.RTROIObservationsSequence.clear()

    for i in range(len(labels)):
        dss = Dataset()
        dss[0x3006, 0x0082] = DataElement(0x30060082, 'IS', str(i + 1))
        dss[0x3006, 0x0084] = DataElement(0x30060084, 'IS', str(i + 1))
        dss[0x3006, 0x0085] = DataElement(0x30060085, 'SH', labels[i])
        dss[0x3006, 0x00a4] = DataElement(0x300600a4, 'CS', '')
        dss[0x3006, 0x00a6] = DataElement(0x300600a6, 'PN', '')

        rs_fp.RTROIObservationsSequence.append(dss)

    #rs_fp.SOPInstanceUID = "{}.{}".format(uid, 1)
    rs_fp.SOPInstanceUID = "{}.{}".format(ct_StudyInstanceUID, 1)
    rs_fp.ApprovalStatus = "UNAPPROVED"


    # ========



    #Start to Interpolate

    make_some_tables(input_ct_filelist)

    #print("input rs filepath for process_label_after_AI = ", Env.FILE_PATH_OF_RS)


    for i in range(len(rs_fp.ROIContourSequence)):
        try:
            item = rs_fp.ROIContourSequence[i].ContourSequence
            organ_exist_z = []
            # organ_exist_item = []
            last_z = 0
            for n in range(len(item)):
                get_z = float(item[n].ContourData[2])
                organ_exist_z.append(get_z)
                last_z = get_z
                # organ_exist_item.append(n)
            # organ_exist_z = np.array(organ_exist_z)
            max_Z = max(organ_exist_z)
            min_Z = min(organ_exist_z)
            print(organ_exist_z)

            thickness = Thickness_table[last_z]  # Actually the thickness from any exist z-slice are the same

            print('thickness = ', thickness)
            organ_table = [i for i in np.arange(min_Z, max_Z + thickness, thickness)]
            print(organ_table)

            for p in range(len(organ_table)):
                z = organ_table[p]
                # print(z)
                # if organ_exist_z.index(z):    #z in organ_exist_z
                if z not in organ_exist_z:  # z in organ_exist_z
                    # print(z)
                    to_fill_z = find_nearest(organ_exist_z, z)
                    to_fill = organ_exist_z.index(to_fill_z)

                    print(z, '___', to_fill_z, to_fill)

                    # cpyItem = copy.deepcopy(rs_fp.ROIContourSequence[i].ContourSequence[to_fill])
                    to_fill_data = rs_fp.ROIContourSequence[i].ContourSequence[to_fill]
                    deepcopy_data = copy.deepcopy(to_fill_data)
                    # rs_fp.ROIContourSequence[i].ContourSequence.append(to_fill_data)
                    rs_fp.ROIContourSequence[i].ContourSequence.append(deepcopy_data)

                    # item.ContourSequence.....append(fill_z)
                    # edit

                    #rs_fp.ROIContourSequence[i].ContourSequence[-1].ContourImageSequence[0].ReferencedSOPInstanceUID = CT_table[to_fill_z]
                    rs_fp.ROIContourSequence[i].ContourSequence[-1].ContourImageSequence[0].ReferencedSOPInstanceUID = CT_table[z]

                    rs_fp.ROIContourSequence[i].ContourSequence[-1].ContourData[2::3] = [z] * (int(len(rs_fp.ROIContourSequence[i].ContourSequence[-1].ContourData) / 3))

        except Exception as e:
            print("Exception")
            traceback.print_tb(e.__traceback__)
            #thickness = Thickness_table[last_z]
            print("last_z = ", last_z)
            print("Thickness_table = ", Thickness_table)


            #print(e)
    #set StructureSetLabel from RALCT_20190411 to RALCT_20190641_AI. The 20190614 is
    studyDate = lastGetDateFromCT
    # Change to 16 bytes because Varian cannot allow 17 bytes
    # rs_fp.StructureSetLabel = "RALCT_{}_AI".format(studyDate)
    rs_fp.StructureSetLabel = "RALCT_{}_A".format(studyDate)

    print("write_file for ", output_rs_filepath_after_ai)
    pydicom.write_file(output_rs_filepath_after_ai, rs_fp)

    return None
