import random
import TSP
import numpy as np
from map import map
import numpy as np
from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update
import math

def angle(a, b, c):
    ab = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    bc = math.sqrt((b[0]-c[0])**2 + (b[1]-c[1])**2 + (b[2]-c[2])**2)
    ac = math.sqrt((a[0]-c[0])**2 + (a[1]-c[1])**2 + (a[2]-c[2])**2)
    return math.acos((ab**2+bc**2-ac**2)/(2*ab*bc))

A = [1, 0, 0]
B = [0, 1, 0]
C = [0, 0, 1]

print(angle(A,B,C))

import math
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def drone_flight(x1, y1, x2, y2,theta):
    dx = x2 - x1
    dy = y2 - y1
    dist = distance(x1, y1, x2, y2)
    angle = math.atan2(dy, dx)
    print("无人机需要飞行", dist, "米")
    print("无人机需要调整方向", math.degrees(angle - theta), "度")
    return dist

x1 = 0
y1 = 0
x2 = 3
y2 = 4
theta = 0
total_distance = distance(x1, y1, x2, y2)

drone_flights = 0
def compute_mu(self,y,drone_distance = 5):
    dist = distance(self[0],self[1],y[0],y[1])
    dx = y[0] - self [0]
    dy = y[1] - self[1]
    angle= math.atan2(dy,dx)
    print("无人机需要调整方向", math.degrees(angle - self[2]), "度")
    times = dist / drone_distance
    # if dist % drone_distance != 0 :
    u = np.zeros((int(times) + 1 ,3))
    # dx = drone_distance * np.cos(angle)
    # dy = drone_distance * np.sin(angle)
    u[:,0] = drone_distance
    u[0,1] = angle - self[2]
    u[-1,0] = dist - drone_distance * int(times)

    self[0]=  y[0]
    self[1] = y[1]
    self[2] = angle
    return  u,self
def compute_path(path,data,self):
    u_out = np.zeros((1,3))
    # print(path,"path_comupute")
    for i in path[0][:]:
        # y[:]= data[i,:]
        # print(data)
        # print(i)
        i = int(i)
        print("第{}个点".format(i))
        y = [data[i,0],data[i,1]]
        u,self = compute_mu(self,y)
        print(self)
        u_out = np.concatenate([u_out,u])
    return u_out


if __name__ == '__main__':
    # data = map(15)[:]
    # print(data.shape)
    # self = [0,0,0.5*np.pi]
    # path = TSP.TSP(data)
    # # print(path.out_put())
    # path = path.out_put()
    # for i in path:
    #     u =  compute_path(i,data,self)
    #     print(u)
    angle = math.atan2(-1, -1)
    print(angle())
    # TSP.draw(data,path)
    # self = [x1,y1,0]
    # y = [x2,y2]
    # print(compute_mu(self,y))



# data = map(15)[:]
# print(data.shape,'datashape')
# TSP.draw(data, TSP.TSP(data))
#
# print(data)




