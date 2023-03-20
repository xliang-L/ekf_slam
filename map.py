import random
import TSP
import numpy as np


class point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, key):
        if key == 1 :
            return self.y
        elif key == 0 :
            return self.x
class drone:
    def __init__(self,x ,y ):

        self.x = x
        self.y = y
        self.v = 0
        self.theta = 0


    def __getitem__(self, key):
        if key == 1:
            return self.y
        elif key == 0:
            return self.x
class map:
    def __init__(self,num = 5):
        self.points = []
        self.num = num
        self.drone = drone(0,0)
        for i in range(num):
            # print(random.Random(0,100),random.Random(0,100))
            po = point(random.random() * 100,random.random() * 100)
            self.points.append(po)
    def __len__(self):
        return self.num + 1

    def __getitem__(self, key):
        if isinstance(key, int):
            if key == 0 :
                return np.array([self.drone[0],self.drone[1]]).reshape(-1, 2)
            else:
                return np.array([self.points[key - 1] [0], self.points[key - 1] [1]]).reshape(-1, 2)
        if isinstance(key, slice):
            start = 0 if key.start == None else key.start
            stop = self.num if key.stop == None else key.stop
            step = 1 if key.step == None else key.step
            # slicedkeys = list(self.points.keys())[key]
            print(start,stop,step)
            list = []
            for i in range(int(start),int(stop),int(step)):
                if i == 0 :
                    list.append(self.drone[0])
                    list.append(self.drone[1])
                    continue

                list.append(self.points[i - 1] [0])
                list.append(self.points[i - 1] [1])
            return np.array(list).reshape(-1,2)

if __name__ == '__main__':
    data = map(50)[:]
    print(data)
    TSP.draw(data, TSP.TSP(data))

    # a = np.array([[4,5,6,1,2,3],[4,5,6,1,2,3]])
    # print(np.where(a == 1))
    # print(a.shape[0])
    # for i in range(a.shape[0]):
    #     si = np.where(a[i] == 1)[0][0]
    #     print(si,'si')
    #     print(a[i])
    #     a[i] = np.concatenate([a[i][si:],a[i][0:si]])
    #     print(a[i])
