#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

scene_points = {
    "bicycle":  int(5900000),
    "flowers":  int(3300000),
    "garden":   int(5200000),
    "stump":    int(4750000),
    "treehill": int(3700000),
    "room":     int(1500000),
    "counter":  int(1200000),
    "kitchen":  int(1800000),
    "bonsai":   int(1300000),

    "chair":    int(300000),
    "drums":    int(300000),
    "ficus":    int(300000),
    "hotdog":   int(300000),
    "lego":     int(300000),
    "materials":int(300000),
    "mic":      int(300000),
    "ship":     int(300000),

    "truck":    int(2600000),
    "train":    int(1100000),

    "drjohnson":int(3400000),
    "playroom": int(2500000),

    # my synthetic
    "buddha_prinBSDF":       int(400000),
    "buddha_spec_prinBSDF":  int(400000),
    "matball":               int(400000),
    "matball_spec":          int(400000),

    # shiny blender synthetic
    "ball":         int(300000),
    "car":          int(300000),
    "coffee":       int(300000),
    "helmet":       int(300000),
    "teapot":       int(300000),
    "toaster":      int(300000),
}
scene_train_cameras = {
    "bicycle":  169,
    "flowers":  151,
    "garden":   161,
    "stump":    109,
    "treehill": 123,
    "room":     272,
    "counter":  210,
    "kitchen":  244,
    "bonsai":   255,

    "chair":    100,
    "drums":    100,
    "ficus":    100,
    "hotdog":   100,
    "lego":     100,
    "materials":100,
    "mic":      100,
    "ship":     100,

    "truck":    219,
    "train":    263,

    "drjohnson":230,
    "playroom": 196,

    "buddha_prinBSDF":      200,
    "buddha_spec_prinBSDF": 200,
    "matball":              200,
    "matball_spec":         200,

    "ball":         100,
    "car":          100,
    "coffee":       100,
    "helmet":       100,
    "teapot":       100,
    "toaster":      100,
}

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
