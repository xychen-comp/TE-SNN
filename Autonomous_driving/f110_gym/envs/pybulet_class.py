import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gym
import numpy as np
import pybullet as p
from gym import logger
from typing import Tuple
import time
import cv2

Position = Tuple[float, float, float] # x, y, z coordinates
Orientation = Tuple[float, float, float] # euler angles
Pose = Tuple[Position, Orientation]
Quaternion = Tuple[float, float, float, float]
Velocity = Tuple[float, float, float, float, float, float]

class Vehicle:
    def __init__(self):
        self._id = None
        self.urdf = 'C:/Users/22038634r/Desktop/f1tenth_race-main/f1tenth_race-main/Autonomous_driving/Autonomous_driving/racecar/racecar.urdf'

    @property
    def id(self) -> Any:
        return self._id
    
    def set_vehicle_pose(self, pose: Pose):
        if not self._id:
            self._id = self._load_model(model=self.urdf, initial_pose=pose)
        else:
            pos, orn = pose
            p.resetBasePositionAndOrientation(self._id, pos, p.getQuaternionFromEuler(orn))
    
    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = p.getQuaternionFromEuler(orientation)
        id = p.loadURDF(model, position, orientation)
        # p.changeVisualShape(id, -1, rgbaColor=self._config.color)
        return id

import os
current_dir = os.path.dirname(os.getcwd())
class World:
    def __init__(self, rendering: bool):
        self._client = None
        self.rendering = rendering
        self.sdf = current_dir+'/Autonomous_driving/f1tenth_racetracks/Barcelona/barcelona.sdf'
        self.vehicle = Vehicle()
        self.init()
        self._up_vector = [0, 0, 1]
        self._camera_vector = [1, 0, 0]
        self._target_distance = 1
        self._fov = 90
        self._near_plane = 0.01
        self._far_plane = 100
        self.width = 87
        self.height = 58

    def init(self) -> None:
        if self.rendering:
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)

        self._load_scene(self.sdf)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(0.01)
        p.setGravity(0, 0, -9.81)

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def set_vehicle_pose(self, pose: Pose):
        self.vehicle.set_vehicle_pose(pose)
        width, height = self.width, self.height
        state = p.getLinkState(self.vehicle.id, linkIndex=5, computeForwardKinematics=True)
        position, orientation = state[0], state[1]
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot(self._camera_vector)
        up_vector = rot_matrix.dot(self._up_vector)
        target = position + self._target_distance * camera_vector
        view_matrix = p.computeViewMatrix(position, target, up_vector)
        aspect_ratio = float(width) / height
        proj_matrix = p.computeProjectionMatrixFOV(self._fov, aspect_ratio, self._near_plane, self._far_plane)
        (_, _, px, depth, _) = p.getCameraImage(width=width,
                                            height=height,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix)

        rgb_array = np.reshape(px, (height, width, -1))
        rgb_array = rgb_array[:, :, :3]
        depth = self._far_plane * self._near_plane / (self._far_plane - (self._far_plane - self._near_plane) * depth)
        depth = np.reshape(depth, (height, width))
        return rgb_array, depth

if __name__ == '__main__':
    world = World(rendering=False)
    start = time.time()
    while True:
        end = time.time()
        pos = (20.0 * np.sin(2 * np.pi * (end - start)), 0, 0.05)
        # pos = (0, 0, 0.05)
        orn = (0, 0, 0)
        pp = (pos, orn)
        rgb, dep = world.set_vehicle_pose(pp)
        dep[dep>6.0] = 6.0
        cv2.imshow("", dep / 6.0)
        key = cv2.waitKey(20)
        if key == 27:
            assert False
        time.sleep(0.01)

