import subprocess
import os
import time

# Set the GPU devices to use
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
gpus = [str(gpu) for gpu in gpus]
# gpus = ['0, 1', '2, 3', '4, 5', '6, 7']

path = "data/mipnerf360"

scenes = ['bicycle', 'kitchen', 'treehill', 'flowers', 'garden', 'bonsai', 'stump', 'room', 'counter']

n_views = [3] * len(scenes)

print(scenes, len(scenes), f'{n_views}')

script = "run.sh"

step = 4

processes = []
available_gpus = gpus[-step:].copy()

try:
    i = 0

    while i < len(scenes) or processes:
        while i < len(scenes) and available_gpus:
            gpu = available_gpus.pop()
            scene = scenes[i]

            args = [scene, str(n_views[i])]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu

            print(f"Starting scene {scene} on GPU {gpu}")
            process = subprocess.Popen(["bash", script] + args, env=env)

            processes.append((process, gpu))

            i += 1

        for process, gpu in processes[:]:
            if process.poll() is not None:
                processes.remove((process, gpu))
                available_gpus.append(gpu)
                print(f"Process for GPU {gpu} finished, GPU is now free.")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nInterrupt received! Terminating running processes...")
    for process, gpu in processes:
        process.terminate()
        print(f"Terminated process on GPU {gpu}")

    print("All processes terminated. Exiting...")
