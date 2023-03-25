# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 22:54:18 2018

@author: Attila
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def stateToArrow(state):
    x = state[0]
    y = state[1]
    dx = 0.5*np.cos(state[2])
    dy = 0.5*np.sin(state[2])
    return x,y,dx,dy

def plotMap(ls,ldt,hist,robot,mapsize):
    plt.clf()
    
    x = robot.x_true
    fov = robot.r
    
    # Plot true environment
    plt.subplot(1,4,1).cla()
    plt.subplot(141, aspect='equal')
    
    # Plot field of view boundaries
    plt.Circle((x[0], x[1]), fov, color='y', fill=False)
    # plt.plot([x[0], x[0]+50*np.cos(x[2] + fov/2)], [x[1], x[1]+50*np.sin(x[2] + fov/2)], color="r")
    # plt.plot([x[0], x[0]+50*np.cos(x[2] - fov/2)], [x[1], x[1]+50*np.sin(x[2] - fov/2)], color="r")
    
    for state in hist:
        plt.arrow(*stateToArrow(state), head_width=0.5)
    plt.scatter(ls[:,0],ls[:,1], s=10, marker="s", color=(0,0,1))
    
    for i in range(ldt.shape[2]):
        plt.scatter(ldt[:,0,i], ldt[:,1,i], s=10, marker="s", color=(0,1,0))
    
    plt.xlim([0,mapsize])
    plt.ylim([0,mapsize])
    plt.title('True environment')
    
    
# Plot:
    # Robot state estimates (red/green)
    # Current robot state covariances
    # Field of view
    # Currently observed landmarks with covariances and lines
    # Previously observed landmarks
    

def plotEstimate(mu, cov, robot, mapsize):
    a = plt.subplot(142, aspect='equal')
    a.cla()
    
    # plot robot state history
    for i in range(mu.shape[1]):
        if i == 0 or i%2 == 1:
            a.arrow(*stateToArrow(mu[:3,i]), head_width=0.5, color=(1,0,0))
        else:
            a.arrow(*stateToArrow(mu[:3,i]), head_width=0.5, color=(0,1,0))
    
    # plot current robot field of view
    fov = robot.r
    # plt.plot([mu[0,-1], mu[0,-1]+50*np.cos(mu[2,-1] + fov/2)], [mu[1,-1], mu[1,-1]+50*np.sin(mu[2,-1] + fov/2)], color="r")
    # plt.plot([mu[0,-1], mu[0,-1]+50*np.cos(mu[2,-1] - fov/2)], [mu[1,-1], mu[1,-1]+50*np.sin(mu[2,-1] - fov/2)], color="r")

    circle =  plt.Circle((mu[0,-1],mu[1,-1]), fov,
                 color="red",fill= False)
    a.add_artist(circle)

    print(mu[0,-1], mu[1,-1])
    # plot current robot state covariance
    robot_cov = Ellipse(xy=mu[:2,-1], width=cov[0,0], height=cov[1,1], angle=0)
    robot_cov.set_edgecolor((0,0,0))
    robot_cov.set_fill(0)
    a.add_artist(robot_cov)
    
    # plot all landmarks ever observed
    n = int((len(mu)-3)/2)
    for i in range(n):
        if cov[2*i+3,2*i+3] < 1e6 and cov[2*i+3,2*i+3] < 1e6:
            zx = mu[2*i+3,-1]
            zy = mu[2*i+4,-1]
            plt.scatter(zx,zy,marker='s', s=10, color=(0,0,1))
    
    # plot settings
    plt.xlim([0,mapsize])
    plt.ylim([0,mapsize])
    plt.title('Observations and trajectory estimate')
    plt.pause(0.1)
    
def plotMeasurement(mu, cov, obs, n,mapsize):

    a = plt.subplot(144, aspect='equal')
    plt.cla()
    plt.xlim([-mapsize/2,mapsize/2])
    plt.ylim([-mapsize/2,mapsize/2])
    for z in obs:
        j = int(z[2])
        zx = mu[2*j+3]
        zy = mu[2*j+4]
        if j < n:
            plt.plot([mu[0][0], zx], [mu[1][0], zy], color=(0,0,1))
        else:
            plt.plot([mu[0][0], zx], [mu[1][0], zy], color=(0,1,0))
        
        landmark_cov = Ellipse(xy=[zx,zy], width=cov[2*j+3][2*j+3], height=cov[2*j+4][2*j+4], angle=0)
        landmark_cov.set_edgecolor((0,0,0))
        landmark_cov.set_fill(0)
        a.add_artist(landmark_cov)
        plt.pause(0.0001)
    plt.xlim([0,mapsize])
    plt.ylim([0,mapsize])
    plt.pause(0.01)

def plotError(mu,x_true):
    # pass

    b = plt.subplot(143)
    mu = mu[:3,0::2] # keep only x,y,theta

    y = (np.asarray(x_true).T)[:,:mu.shape[1]]
    print(mu.shape)
    # print(x_true.shape)
    # mu = mu[:,:x_true.shape[1]]
    try:
        dif = np.power(np.abs(mu - y),2)
        err = dif[0,:] + dif[1,:]
        b.plot(err, color="r")
        plt.title('Squared estimation error')
        plt.xlabel('Steps')
        plt.ylabel('Squared error')
    except:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#    b.plot(dif[2,:])