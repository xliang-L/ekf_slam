# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:14:20 2018

@author: Attila
"""

import numpy as np

class Robot:
    def __init__(self, x_init, r , Rt, Qt):
        x_init[2] = (x_init[2]+np.pi)%(2*np.pi)-np.pi

        self.x_true = x_init
        
        self.lo = np.empty((0,3))
        self.r =  r
        
        # noise covariances
        self.Rt = Rt
        self.Qt = Qt
    
    def move(self,u):
        # Make noisy movement in environment
        # motion_noise = np.matmul(np.random.randn(1,3),self.Rt)[0]
        # [dtrans, drot1, drot2] = u[:3] + motion_noise
        [dtrans, drot1, drot2] = u[:3]
        x = self.x_true
        x_new = x[0] + dtrans*np.cos(x[2]+drot1)
        y_new = x[1] + dtrans*np.sin(x[2]+drot1)
        theta_new = (x[2] + drot1 + drot2 + np.pi) % (2*np.pi) - np.pi

        self.x_true = [x_new, y_new, theta_new]

        return self.x_true

        # motion_noise = np.matmul(np.random.randn(1,3),self.Rt)[0]z
        # d  = u[:3] + motion_noise
        # self.x_true += d
        return  self.x_true



    
    def sense(self,lt):
        # Make noisy observation of subset of landmarks in field of view
        
        x = self.x_true
        observation = np.empty((0,3))

        # fovL = (x[2]+self.fov/2+2*np.pi)%(2*np.pi)
        # fovR = (x[2]-self.fov/2+2*np.pi)%(2*np.pi)
        
        for landmark in lt:
            meas_range = np.sqrt(np.power(landmark[1] - x[1], 2) + np.power(landmark[0] - x[0], 2)) + self.Qt[0][
                0] * np.random.randn(1)
            rel_angle = np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))
            rel_angle_2pi = (np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))+2*np.pi)%(2*np.pi)
            # TODO: re-include and debug field of view constraints
            # if (fovL - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi > 0 and (fovR - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi < 0:
            if self.r > meas_range :
                meas_range = np.sqrt(np.power(landmark[1]-x[1],2)+np.power(landmark[0]-x[0],2)) + self.Qt[0][0]*np.random.randn(1)
                meas_bearing = (rel_angle - x[2] + self.Qt[1][1]*np.random.randn(1) + np.pi)%(2*np.pi)-np.pi
                observation = np.append(observation,[[meas_range[0], meas_bearing[0], landmark[2]]],axis=0)
                
        return observation


class drone:
    def __init__(self, x_init, r, Rt, Qt):
        x_init[2] = (x_init[2] + np.pi) % (2 * np.pi) - np.pi

        self.x_true = x_init

        self.lo = np.empty((0, 3))
        self.r = r

        # noise covariances
        self.Rt = Rt
        self.Qt = Qt

    def move(self, u):
        # Make noisy movement in environment
        motion_noise = np.matmul(np.random.randn(1, 3), self.Rt)[0]
        [dtrans, drot1, drot2] = u[:3] + motion_noise

        x = self.x_true
        x_new = x[0] + dtrans * np.cos(x[2] + drot1)
        y_new = x[1] + dtrans * np.sin(x[2] + drot1)
        theta_new = (x[2] + drot1 + drot2 + np.pi) % (2 * np.pi) - np.pi

        self.x_true = [x_new, y_new, theta_new]

        return self.x_true

    def sense(self, lt):
        # Make noisy observation of subset of landmarks in field of view

        x = self.x_true
        observation = np.empty((0, 3))

        # fovL = (x[2]+self.fov/2+2*np.pi)%(2*np.pi)
        # fovR = (x[2]-self.fov/2+2*np.pi)%(2*np.pi)

        for landmark in lt:
            meas_range = np.sqrt(np.power(landmark[1] - x[1], 2) + np.power(landmark[0] - x[0], 2)) + self.Qt[0][
                0] * np.random.randn(1)
            rel_angle = np.arctan2((landmark[1] - x[1]), (landmark[0] - x[0]))
            rel_angle_2pi = (np.arctan2((landmark[1] - x[1]), (landmark[0] - x[0])) + 2 * np.pi) % (2 * np.pi)
            # TODO: re-include and debug field of view constraints
            # if (fovL - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi > 0 and (fovR - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi < 0:
            if self.r > meas_range:
                meas_range = np.sqrt(np.power(landmark[1] - x[1], 2) + np.power(landmark[0] - x[0], 2)) + self.Qt[0][
                    0] * np.random.randn(1)
                meas_bearing = (rel_angle - x[2] + self.Qt[1][1] * np.random.randn(1) + np.pi) % (2 * np.pi) - np.pi
                observation = np.append(observation, [[meas_range[0], meas_bearing[0], landmark[2]]], axis=0)

        return observation

    def __getitem__(self, key):
        if key == 1:
            return self.y
        elif key == 0:
            return self.x