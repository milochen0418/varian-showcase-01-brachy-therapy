import os
import datetime
import AI_process as ai
import SimpleInterpolateRsWrapUp as sirw

def log_current_time(tag_str, status):
    timefmt = "%Y-%m%d-%H:%M:%S.%f"
    now = datetime.datetime.now()
    nowtime_str = now.strftime(timefmt)
    print("[{}/{}] = {}".format( tag_str, status,  nowtime_str ))




if __name__ == "__main__":

    model_name = "MRCNN_Brachy"
    ct_folder = "TestCase_Input_CtFolder"

    ct_filelist = []
    for file in os.listdir(ct_folder):
        filepath = "{}\\{}".format(ct_folder, file)
        ct_filelist.append(filepath)

    print("len of ct_filelist = ", len(ct_filelist))

    log_current_time("AI_process", "START")
    # mrcnn_out = ai.AI_process_by_folder(ct_folder, model_name)
    mrcnn_out = ai.AI_process(ct_filelist, model_name)
    log_current_time("AI_process", "STOP")

    #print(mrcnn_out)

    log_current_time("InterpolateWrapper_process", "START")
    sirw.interpolate_and_wrapup_rs(mrcnn_out, ct_filelist, "RS.output.dcm")
    log_current_time("InterpolateWrapper_process", "STOP")

