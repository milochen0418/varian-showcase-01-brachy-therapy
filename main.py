import os
import datetime
import AI_process as ai
import SimpleInterpolateRsWrapUp as sirw
from SimpleInterpolateRsWrapUp import interpolate_and_wrapup_rs
from AI_process import AI_process_get_predict_result

def log_current_time(tag_str, status):
    timefmt = "%Y-%m%d-%H:%M:%S.%f"
    now = datetime.datetime.now()
    nowtime_str = now.strftime(timefmt)
    print("[{}/{}] = {}".format( tag_str, status,  nowtime_str ))

def generate_rs_by_ct_folder(input_ct_folder, output_rs_filepath, model_name):
    import pydicom
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

def generate_rp_by_folder(folder):
    """

    :param folder:
        The folder should contain RS file and CT files
    :return:
    """

if __name__ == "__main__":
    model_name = "MRCNN_Brachy"
    input_folder = "TestCase_Input_CtFolder"
    output_folder = "OutputFolder"

    generate_rs_by_ct_folder(input_ct_folder = input_folder, output_rs_filepath='RS.ggg.dcm', model_name = "MRCNN_Brachy")
    exit()


