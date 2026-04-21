import subprocess
import os
from glob import glob
import time

# Set the GPU devices to use (e.g. 0,1,2,3)
gpus = [0, 1, 2, 3, 4, 5, 6, 7][:6]#[1:]#[1:]
# gpus = [0, 2, 4, 5, 6, 7]
# gpus = [3, 4, 6, 7]
gpus = [str(gpu) for gpu in gpus]
# gpus = ['0, 1', '2, 3', '4, 5', '6, 7']#[1:]

path = "data/co3d"
# index = 1
# image_name = "_DSC8707"


scenes = [f'{x.split("/")[-2]}/{x.split("/")[-1]}' for x in glob(f"{path}/*/*")]


n_views = [3] * len(scenes) + [6] * len(scenes) + [9] * len(scenes)
scenes = sorted(scenes) * 3

# n_views = [6] * len(scenes) + [9] * len(scenes)
# scenes = sorted(scenes) * 2


# n_views = [3] * len(scenes)
# scenes = sorted(scenes)

print(scenes, len(scenes), f'{n_views}')
# scenes = scenes[:1]
# scenes = ['bench/167_18175_33897', 'bench/185_19987_38634']
# exit()

# path = '/home/grads/a/avinashpaliwal/github/GaussianObject/data/mipnerf360'

# scenes = sorted(os.listdir(path))

# # scenes = ['treehill', 'garden', 'bonsai', 'bicycle', 'flowers', 'room', 'kitchen', 'counter']#[:4]

# scenes = ['treehill', 'garden', 'bonsai', 'stump', 'flowers', 'room', 'kitchen', 'counter']#[:4]
# # scenes = ['counter', 'bicycle', 'stump']
# # scenes = ['stump']
# # scenes = ['stump', 'counter', 'bicycle']
# # scenes = ['bicycle', 'treehill', 'garden'] #+ ['room', 'kitchen', 'bonsai']
# # 
# # scenes = ['bicycle', 'treehill', 'flowers']
# # scenes = ['bonsai', 'kitchen', 'room']
# # scenes = ['stump', 'counter', 'garden']
# # scenes = ['stump', 'counter']

# # scenes = ['treehill', 'stump', 'bicycle', 'flowers', 'garden']
# # scenes = ['treehill', 'stump', 'bicycle', 'garden']
# # scenes = ['bonsai', 'room', 'kitchen', 'counter', 'flowers']

# Set the Python script to run
script = "eval.sh"

# step = 7
# step = 8
step = 6


# Queue of processes
processes = []
available_gpus = gpus[-step:].copy()

print(available_gpus)
# exit()

try:
    i = 0

    while i < len(scenes) or processes:
        # Start new processes if there are available GPUs and scenes left
        while i < len(scenes) and available_gpus:
            gpu = available_gpus.pop()
            scene = scenes[i]
            
            # Set the CUDA_VISIBLE_DEVICES environment variable
            args = [scene, str(n_views[i])]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            
            # Run the script with the specified GPU
            print(f"Starting scene {scene} on GPU {gpu}")
            process = subprocess.Popen(["bash", script] + args, env=env)
            
            # Append process and associated GPU to track
            processes.append((process, gpu))
            
            i += 1

        # Check if any process has finished
        for process, gpu in processes[:]:
            if process.poll() is not None:  # Process finished
                processes.remove((process, gpu))
                available_gpus.append(gpu)  # GPU becomes available
                print(f"Process for GPU {gpu} finished, GPU is now free.")
        
        # Slight delay to avoid busy-waiting (optional)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nInterrupt received! Terminating running processes...")
    # Terminate all running processes
    for process, gpu in processes:
        process.terminate()
        print(f"Terminated process on GPU {gpu}")
    
    print("All processes terminated. Exiting...")
# for i in [0, step, step*2]:
# # for i in [0, 3, 6, 9, 12, 15, 18]:
# # for i in [7, 14]:

#     print("Running scenes", i, i+step, scenes[i:i+step])
#     # scenes = 
#     # Loop through each GPU and run the script in parallel
#     processes = []
#     for scene, gpu in zip(scenes[i:i+step], gpus[-step:]):
#         # Set the CUDA_VISIBLE_DEVICES environment variable
#         # Set the arguments for the script (if any)
#         args = [scene]
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = gpu#str(gpu)
        
#         # Run the script with the specified GPU
#         print("bash", script, args, gpu)
#         process = subprocess.Popen(["bash", script] + args, env=env)
#         processes.append(process)

#     # Wait for all processes to finish
#     for process in processes:
#         process.wait()
    
#     # break


# #     args = [scene]
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
# #     # Run the script with the specified GPU
# #     print("bash", script, args, gpu)
# #     process = subprocess.Popen(["bash", script] + args, env=env)