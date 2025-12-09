import os
import shutil

from sbto.data.utils import get_config_dict_from_rundir, get_filename_from_path

def group_run_dir_by_ref_file_name(task_dir: str):
    """
    Group run directories by reference file names.
    To be used when generating from a lot of reference files. 
    """
    for dir in os.listdir(task_dir):
        run_dir = os.path.join(task_dir, dir)
        if os.path.isdir(run_dir):
            cfg = get_config_dict_from_rundir(run_dir)
            # Get all ref file paths
            try:
                ref_motion_path = cfg["task"]["cfg_ref"]["motion_path"]
            except Exception as e:
                continue
            
            # Create new dir with same name as motion ref
            # Move the rundir data there
            ref_motion_name = get_filename_from_path(ref_motion_path)
            run_dir_dst = os.path.join(task_dir, ref_motion_name)
            os.makedirs(run_dir_dst, exist_ok=True)
            shutil.move(run_dir, run_dir_dst)