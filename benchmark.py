import os
import sys
import time

from argparse import ArgumentParser
from scene import scene_points, scene_train_cameras

ratio = 0.1 # Ratio of additional gaussians to the original number of gaussians.

##################################################################################

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
nerf_synthetic_scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
tanks_and_temples_scenes = ["train", "truck"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Program may fail to save point cloud files(.ply) on windows, so we add retry mechanism.
def run_command_with_retry(cmd, max_retries=3):
    for attempt in range(max_retries):
        start_time = time.time()
        result = os.system(cmd)
        if result == 0:
            return True, time.time() - start_time
        print(f"The command '{cmd}' failed with exit code {result}. Retrying...")
        time.sleep(3)
    print(f"Failed to execute command after {max_retries} attempts: {cmd}")
    return False, -1.0

def main(args):
    python_path = sys.executable

    datasets_dir = {
        'mipnerf360': args.m360,
        'nerf_synthetic': args.ns,
        'tanks_and_temples': args.tnt,
        'deep_blending': args.db,
    }

    scenes = []
    if args.ns:
        scenes.extend(nerf_synthetic_scenes)
    if args.m360:
        scenes.extend(mipnerf360_outdoor_scenes)
        scenes.extend(mipnerf360_indoor_scenes)
    if args.tnt:
        scenes.extend(tanks_and_temples_scenes)
    if args.db:
        scenes.extend(deep_blending_scenes)                                                                                                                   

    logging = {}
    for sn in scenes:
        logging[sn] = {}

    # Training 3dgs / 3dgs-mcmc
    if not args.skip_training_3dgs:
        common_args = f" --disable_viewer --quiet --eval --test_iterations -1 --save_iterations 30000 --sh_degree {args.sh_degree} --seed {args.seed} "
        if args.mcmc:
            script = "train_mcmc.py"
            common_args += " --opacity_lr 0.05 --densify_until_iter 25000 "
        else:
            script = "train_3dgs.py"

        for sn in scenes:
            capacity = int(scene_points[sn] / (ratio + 1.0))
            if sn in mipnerf360_outdoor_scenes:
                _, elpased_time = run_command_with_retry(" ".join([
                    python_path, script, f"-s {datasets_dir['mipnerf360']}/{sn}",
                    f"-m output/MipNerf360/{sn}", "--resolution 4", common_args,
                    f"--cap_max {capacity}" if args.mcmc else ""
                ]))
            elif sn in mipnerf360_indoor_scenes:
                _, elpased_time = run_command_with_retry(" ".join([
                    python_path, script, f"-s {datasets_dir['mipnerf360']}/{sn}",
                    f"-m output/MipNerf360/{sn}", "--resolution 2", common_args,
                    f"--cap_max {capacity}" if args.mcmc else ""
                ]))
            elif sn in nerf_synthetic_scenes:
                _, elpased_time = run_command_with_retry(" ".join([
                    python_path, script, f"-s {datasets_dir['nerf_synthetic']}/{sn}",
                    f"-m output/nerf_synthetic/{sn}", common_args,
                    f"--cap_max {capacity}" if args.mcmc else "",
                ]))
            elif sn in tanks_and_temples_scenes:
                _, elpased_time = run_command_with_retry(" ".join([
                    python_path, script, f"-s {datasets_dir['tanks_and_temples']}/{sn}",
                    f"-m output/tanks_and_temples/{sn}", common_args,
                    f"--cap_max {capacity}" if args.mcmc else ""
                ]))
            elif sn in deep_blending_scenes:
                _, elpased_time = run_command_with_retry(" ".join([
                    python_path, script, f"-s {datasets_dir['deep_blending']}/{sn}",
                    f"-m output/deep_blending/{sn}", common_args,
                    f"--cap_max {capacity}" if args.mcmc else "",
                    "--opacity_reg 0.001" if sn == "drjohnson" else "",
                ]))
            else:
                raise ValueError(f"Unknown scene: {sn}")
            
            logging[sn]["train_3dgs"] = elpased_time
        
        print(logging)

    ################################################################################

    for sn in scenes:
        if not args.skip_projecting_2d:            
            if (sn in mipnerf360_outdoor_scenes) or (sn in mipnerf360_indoor_scenes):
                output_path = f"output/MipNerf360/{sn}"
            if sn in nerf_synthetic_scenes:
                output_path = f"output/nerf_synthetic/{sn}"
            if sn in tanks_and_temples_scenes:
                output_path = f"output/tanks_and_temples/{sn}"
            if sn in deep_blending_scenes:
                output_path = f"output/deep_blending/{sn}"
        
            time.sleep(3)
            _, elpased_time = run_command_with_retry(" ".join([
                python_path, "train_2d.py",
                f"-m {output_path}",
                f"--seed {args.seed}",
                f"--ratio {ratio}",
                "--skip_train_2d" if args.skip_training_2d else " "
            ]))
            logging[sn]["train_2d"] = elpased_time

        if (sn in mipnerf360_indoor_scenes) or (sn in mipnerf360_outdoor_scenes):
            source_path = f"{datasets_dir['mipnerf360']}/{sn}"
            output_path = f"output/MipNerf360/{sn}"
        elif sn in nerf_synthetic_scenes:
            source_path = f"{datasets_dir['nerf_synthetic']}/{sn}"
            output_path = f"output/nerf_synthetic/{sn}"
        elif sn in tanks_and_temples_scenes:
            source_path = f"{datasets_dir['tanks_and_temples']}/{sn}"
            output_path = f"output/tanks_and_temples/{sn}"
        elif sn in deep_blending_scenes:
            source_path = f"{datasets_dir['deep_blending']}/{sn}"
            output_path = f"output/deep_blending/{sn}"

        train_cameras_count = scene_train_cameras[sn]
        save_iterations = " ".join([str(30000 + train_cameras_count * p) for p in [30]])
        iterations = int(save_iterations.split()[-1])

        if not args.skip_refining_3d:
            train_args = " ".join([
                    f"--source_path {source_path}",
                    f"-m {output_path}",
                    "--disable_viewer", f"--seed {args.seed}",
                    f"--start_checkpoint", f"{output_path}/point_cloud/iteration_30000/aug_point_cloud_init.ply",
                    "--eval",
                    f"--save_iterations {save_iterations}",
                    # f"--test_iterations 30000 {save_iterations}",
                    f"--iterations {iterations}",
                    f"--sh_degree {args.sh_degree}",
                    "--position_lr_init", "0.000016",
                    "--position_lr_final", "0.000016",
                    "--feature_lr", "0.001",
                    "--opacity_lr", "0.02",
                    "--scaling_lr", "0.002",
                    "--rotation_lr", "0.0005",
                    "--opadir_lr", "0.001",
                    "--theta_lr", "0.0002",
                    "--beta_lr", "0.002",
                ])
            
            time.sleep(3)
            if sn in mipnerf360_outdoor_scenes:
                _, elpased_time = run_command_with_retry(python_path + " refine.py " + "--resolution 4 " + train_args)
            elif sn in mipnerf360_indoor_scenes:
                _, elpased_time = run_command_with_retry(python_path + " refine.py " + "--resolution 2 " + train_args)
            else:
                _, elpased_time = run_command_with_retry(python_path + " refine.py " + train_args)
            logging[sn]["refine_3d"] = elpased_time

        # Render & Metric
        if not args.skip_metrics:
            time.sleep(3)
            os.system(" ".join([
                python_path, "render.py",
                f"--source_path {source_path}",
                f"-m {output_path}",
                "--eval",
                "--skip_train",
                f"--iterations {save_iterations}"
            ]))

    # Save logging
    print(logging)
    with open("benchmark_log.txt", "w") as f:
        for sn in scenes:
            f.write(f"Scene: {sn}\n")
            for key in logging[sn]:
                f.write(f"  {key}: {logging[sn][key]:.2f} sec\n")
            f.write("\n")
    
    return

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--mcmc", action="store_true")

    parser.add_argument("--skip_training_3dgs", action="store_true", help="Skip training 3DGS")
    parser.add_argument("--skip_training_2d", action="store_true", help="Skip training 2D")
    parser.add_argument("--skip_projecting_2d", action="store_true", help="Skip projecting to world")
    parser.add_argument("--skip_refining_3d", action="store_true", help="Skip refining 3DGS")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip rendering and metrics")

    parser.add_argument("-m360", type=str, help="Path to MipNeRF360 dataset")
    parser.add_argument("-ns", type=str, help="Path to NeRF Synthetic dataset")
    parser.add_argument("-tnt", type=str, help="Path to Tanks and Temples dataset")
    parser.add_argument("-db", type=str, help="Path to Deep Blending dataset")

    args = parser.parse_args(sys.argv[1:])

    main(args)