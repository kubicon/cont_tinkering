
def prepare_sbatch_part(time, memory, gpu = False):
  sbatch_part = "#!/bin/bash\n"
  if gpu:
    if time <= 4:
      partition = "amdgpufast"
    elif time <= 24:
      partition = "amdgpu"
    elif time <= 72:
      partition = "amdgpulong"
    else:
      assert False, "Time too long"
  else:
    if time <= 4:
      partition = "amdfast"
    elif time <= 24:
      partition = "amd"
    elif time <= 72:
      partition = "amdlong"
    else:
      assert False, "Time too long"
  sbatch_part += f"#SBATCH --partition={partition}\n"
  sbatch_part += f"#SBATCH --time={time}:00:00\n"
  sbatch_part += f"#SBATCH --mem={memory}G\n"
  if gpu:
    sbatch_part += f"#SBATCH --gres=gpu:1\n"
  return sbatch_part

def prepare_libraries():
  library_string = ""
  library_string += f"ml Python/3.13.5-GCCcore-14.3.0 \n"
  library_string += f"ml CUDA/12.9.1 \n" 
  library_string += f"ml cuDNN/9.13.1.26-CUDA-12.9.1 \n"
  library_string += f"ml NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1 \n"
  library_string += f"ml SciPy-bundle/2025.07-gfbf-2025b  \n"
  library_string += f"ml typing-extensions/4.14.1-GCCcore-14.3.0 \n"
  library_string += f"ml matplotlib/3.10.5-gfbf-2025b \n"
  library_string += f"ml PyYAML/6.0.2-GCCcore-14.3.0 \n"
  library_string += f"source /home/kubicon3/mnt_personal/git/cont_tinkering/.venv/bin/activate\n"
  return library_string


def prepare_default_script(time, memory, gpu = False):
  return prepare_sbatch_part(time, memory, gpu) + prepare_libraries()