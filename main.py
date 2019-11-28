import os
import datetime
def log_current_time(tag_str, status):
    timefmt = "%Y-%m%d-%H:%M:%S.%f"
    now = datetime.datetime.now()
    nowtime_str = now.strftime(timefmt)
    print("[{}/{}] = {}".format( tag_str, status,  nowtime_str ))
def generate_rs_by_ct_folder(input_ct_folder, output_rs_filepath, model_name):
    """
    :param input_ct_folder:
    :param output_rs_filepath:
    :param model_name:
    :return:
    """
    from SimpleInterpolateRsWrapUp import interpolate_and_wrapup_rs
    from AI_process import AI_process_get_predict_result
    import pydicom
    ct_folder = input_ct_folder
    ct_filelist = []
    for file in os.listdir(ct_folder):
        filepath = os.path.join(ct_folder, file)
        try:
            ct_fp = pydicom.read_file(filepath)
            if ct_fp.Modality == 'CT':
                ct_filelist.append(filepath)
        except:
            pass
    print("len of ct_filelist = ", len(ct_filelist))
    log_current_time("AI_process", "START")
    # mrcnn_out = ai.AI_process_by_folder(ct_folder, model_name)
    #mrcnn_out = ai.AI_process(ct_filelist, model_name)
    mrcnn_out = AI_process_get_predict_result(ct_filelist, model_name)
    log_current_time("AI_process", "STOP")
    #print(mrcnn_out)

    log_current_time("InterpolateWrapper_process", "START")
    #sirw.interpolate_and_wrapup_rs(mrcnn_out, ct_filelist, "RS.output.dcm")
    interpolate_and_wrapup_rs(mrcnn_out, ct_filelist, "RS.output.dcm")
    log_current_time("InterpolateWrapper_process", "STOP")
def generate_rp_by_ct_rs_folder(input_ct_rs_folder, output_rp_filepath):
    from utilities import generate_metadata_to_dicom_dict
    from generate_rp_brachy_in_batch import get_dicom_dict
    from generate_rp_brachy_in_batch import generate_brachy_rp_file
    from generate_rp_brachy_in_batch import generate_output_to_dicom_dict
    folder = input_ct_rs_folder
    dicom_dict = get_dicom_dict(folder)
    generate_metadata_to_dicom_dict(dicom_dict)
    generate_output_to_dicom_dict(dicom_dict)
    generate_brachy_rp_file(RP_OperatorsName='cylin',
                            dicom_dict=dicom_dict,
                            out_rp_filepath=output_rp_filepath,
                            is_enable_print=False)
def generate_rs_rp_by_ct_folder(input_ct_folder, output_rs_rp_folder, model_name):
    def clean_all_files_in_folder(folder):
        import shutil
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (filepath, e))
    from utilities import create_directory_if_not_exists
    from shutil import copyfile
    import pydicom
    model_name = "MRCNN_Brachy"
    input_folder = "TestCase_Input_CtFolder"
    output_folder = "OutputFolder"
    temp_folder = r"temp"
    create_directory_if_not_exists(temp_folder)
    clean_all_files_in_folder(temp_folder)
    # Step 1. copy all ct file from input folder to temp folder
    for file in os.listdir(input_folder):
        filepath = os.path.join(input_folder, file)
        try:
            fp = pydicom.read_file(filepath)
            if fp.Modality == 'CT':
                src_ct_filepath = filepath
                dst_ct_filepath = os.path.join(temp_folder, os.path.basename(filepath))
                copyfile(src_ct_filepath, dst_ct_filepath)
        except:
            pass
    # Step 2. gen rs in the temp folder
    generate_rs_by_ct_folder(
        input_ct_folder=temp_folder,
        output_rs_filepath=os.path.join(temp_folder, r'RS.output.dcm'),
        model_name="MRCNN_Brachy"
    )

    # Step 3. gen rp in the temp folder
    generate_rp_by_ct_rs_folder(
        input_ct_rs_folder=temp_folder,
        output_rp_filepath=os.path.join(temp_folder, r"RP.output.dcm")
    )

    # Step 4. copy rs and rp file into output_folder
    print("temp_folder = ".format(temp_folder))
    for file in os.listdir(temp_folder):
        filepath = os.path.join(temp_folder, file)
        try:
            fp = pydicom.read_file(filepath)
            if fp.Modality == "RTDOSE":
                rd_src_filepath = filepath
                rd_dst_filepath = os.path.join(output_folder, os.path.join(output_folder, os.path.basename(rd_src_filepath)))
                copyfile(rd_src_filepath, rd_dst_filepath)
            elif fp.Modality == "RTSTRUCT":
                rs_src_filepath = filepath
                rs_dst_filepath = os.path.join(output_folder, os.path.join(output_folder, os.path.basename(rs_src_filepath)))
            copyfile(rs_src_filepath, rs_dst_filepath)
        except:
            pass



def dev_test_code_running():
    # example code of how to gen RS from CT folder
    def example_of_gen_rs():
        model_name = "MRCNN_Brachy"
        input_folder = "TestCase_Input_CtFolder"
        output_folder = "OutputFolder"

        generate_rs_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_filepath=os.path.join(input_folder, r'RS.output.dcm'),
            model_name="MRCNN_Brachy")
    # example code of how to gen RP from CT RS folder
    def example_of_gen_rp():
        generate_rp_by_ct_rs_folder(
            input_ct_rs_folder=r"RAL_plan_new_20190905\29059811-1",
            output_rp_filepath=r"RP.output.dcm")
    # example code of how to gen RS & RP from CT folder
    def example_of_gen_rs_rp():
        input_folder = r"ShowCase01Test-Input-29059811"
        output_folder = r"ShowCase01Test-Output-29059811"
        generate_rs_rp_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_rp_folder=output_folder,
            model_name="MRCNN_Brachy")

    example_of_gen_rs_rp()
    pass


if __name__ == "__main__":
    #example_of_gen_rs()
    dev_test_code_running()
    exit()


