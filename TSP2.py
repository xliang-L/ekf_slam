from math import floor
import numpy as np
import matplotlib.pyplot as plt  # 导入所需要的库
from map import map

class Gena_TSP(object):
    def __init__(self, data,self_num,matrix_distance,index, maxgen=500, size_pop=200, cross_prob=0.9, pmuta_prob=0.01, select_prob=0.8,distance = 300):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数 200
        self.cross_prob = cross_prob  # 交叉概率 0.9
        self.pmuta_prob = pmuta_prob  # 变异概率0.01
        self.select_prob = select_prob  # 选择概率 0.8
        self.distance = distance
        self.data = data  # 城市的左边数据
        self.num = self_num  # 城市个数 对应染色体长度
        self.index = np.array(index)
        # print(self.index.any(),'sdklfjaklsdflkajsdfjlkasdf')
        self.matrix_distance = matrix_distance
        # 距离矩阵n*n, 第[i,j]个元素表示城市i到j距离matrix_dis函数见下文

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        # 通过选择概率确定子代的选择个数    160

        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)   # 200 * 15
        self.sub_sel = np.array([0] * self.select_num * self.num).reshape(self.select_num, self.num) # 160 * 15
        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）

        self.fitness = np.zeros(self.size_pop) # 200  * 1
        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数

        self.best_fit = []
        self.best_path = []
        # 保存每一步的群体的最优路径和距离

    # 随机生成的部分
    def rand_chrom(self):
        if self.index.any() == None :
            print('xxxxxxxxxxxxxxx')
            rand_ch = np.array(range(self.num))
            for i in range(self.size_pop):
                np.random.shuffle(rand_ch)
                self.chrom[i,:]= rand_ch
                self.fitness[i] = self.comp_fit(rand_ch)
        else:
            print('yyyyyyyyyyyyyyy')
            rand_ch = self.index
            for i in range(self.size_pop):
                # new_ch = self.init_sub_child(rand_ch)
                self.chrom[i,:] = rand_ch
                self.fitness[i] = self.comp_fit(rand_ch)
    # def init_sub_child(self,ch):
    #     r1 = np.random.randint(self.num)
    #     r2 = np.random.randint(self.num)
    #     while r2 == r1:
    #         r2 = np.random.randint(self.num)
    #     left, right = min(r1, r2), max(r1, r2)
    #     print(left,right,ch)
    #     np.random.shuffle(ch[left:right])
    #     return ch




    #  计算适应度
    def comp_fit(self, one_path):
        b = np.where(one_path == 0)[0][0]
        # one_path = np.concatenate(one_path)
        one_path =np.concatenate([one_path[b:],one_path[0:b]])
        res = 0

        dis = self.distance
        for i in range(self.num-1):
            if dis > self.matrix_distance[one_path[i],one_path[i+1]] + self.matrix_distance[one_path[i+ 1],one_path[0]] :
                res += self.matrix_distance[one_path[i],one_path[i+1]]
                dis -= self.matrix_distance[one_path[i],one_path[i+1]]
            else:
                res += self.matrix_distance[one_path[i],one_path[0]]
                res += self.matrix_distance[one_path[0],one_path[i+1]]
                res += dis
                dis = self.distance
                dis -= self.matrix_distance[one_path[i+ 1],one_path[0]]

        res += self.matrix_distance[one_path[-1],one_path[0]]
        return res

    # 输出一条路径
    def out_path(self, one_path):
        b = np.where(one_path == 0)[0][0]
        # one_path = np.concatenate(one_path)
        one_path =np.concatenate([one_path[b:],one_path[0:b]])

        res = str(one_path[0]+1)+'-->'
        for i in range(1, self.num):
            res += str(one_path[i]+1)+'-->'
        res += str(one_path[0]+1)+'\n'
        print(res)

    # 轮盘赌  应该换成 锦标赛试一试
    def select_sub(self):
        fit = 1./(self.fitness) #适应度函数
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1]/self.select_num *(np.random.rand()+np.array(range(self.select_num)))
        i,j = 0,0
        index = []
        while i<self.size_pop and j<self.select_num:
            if cumsum_fit[i]>= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index,:]
        # for
            # print(self.sub_sel[i])

        # print(self.sub_sel.shape,'sub_sel')

    #     输出路径
    def out_put(self):
        one_path = self.chrom[0]
        print(one_path, 'one_paht')
        b = np.where(one_path == 0)[0][0]
        # one_path = np.concatenate(one_path)
        one_path =np.concatenate([one_path[b:],one_path[0:b]])
        print(one_path,'one_paht')
        dis = self.distance
        path = np.array(0)
        list = []
        j = 0
        for i in range(self.num-1):
            if dis > self.matrix_distance[one_path[i],one_path[i+1]] + self.matrix_distance[one_path[i+ 1],one_path[0]] :
                dis -= self.matrix_distance[one_path[i],one_path[i+1]]
            else:

                # print(one_path[0:i], np.array(one_path[0]).reshape(-1))
                if j == 0:
                    path  = np.concatenate([one_path[j:i],np.array(one_path[0]).reshape(-1)])
                else:
                    path  = np.concatenate([np.array(one_path[0]).reshape(-1),one_path[j:i],np.array(one_path[0]).reshape(-1)])
                j = i
                list.append([path,dis])
                dis = self.distance
                dis -= self.matrix_distance[one_path[i+ 1],np.array(one_path[0]).reshape(-1)]
                # print(one_path[0:i],path)
        if j == 0:
            path  = np.concatenate([one_path[j:self.num],np.array(one_path[0]).reshape(-1)])
        else:
            path  = np.concatenate([np.array(one_path[0]).reshape(-1),one_path[j:self.num],np.array(one_path[0]).reshape(-1)])
        # path = np.concatenate([one_path[j:self.num], np.array(one_path[0]).reshape(-1)])
        list.append([path,dis])
        return  list

    # 交叉
    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num - 1, 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    #             变异部分
    def intercross(self, ind_a, ind_b):
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b
    def mutation_sub(self):
        for i in range(self.select_num):
            if np.random.rand() <= self.cross_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.sub_sel[i,[r1,r2]] = self.sub_sel[i,[r2,r1]]
    def reverse_sub(self):
        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            left, right = min(r1, r2), max(r1, r2)
            sel = self.sub_sel[i, :].copy()

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):
                self.sub_sel[i, :] = sel
    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num],:] = self.sub_sel

        # for i in self.chrom:

        # print(type(self.chrom))









def draw(data,Path_short):
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(x, y, linewidths=0.1)
    for i, txt in enumerate(range(1, len(data) + 1)):
        ax.annotate(txt, (x[i], y[i]))
    # res0 = Path_short.chrom[0]
    res = Path_short.out_put()
    cl = ['r','b','g','c','m','y','k']
    for i in range(len(res)):
        color = cl[i%7]
        res0 = res[i][0]
        print("第{}个路线剩余的distance = {}".format(i+1,res[i][1]))

        # print(re)
        x0 = x[res0]
        y0 = y[res0]
        for i in range(len(res0) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color=color, width=0.001, angles='xy', scale=1,
                       scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color=color, width=0.001, angles='xy', scale=1,
                   scale_units='xy')
    plt.show()
    # print('路程: ' + str(Path_short.fitness[0]))


def TSP(data,self_num,matrix_distance,index,maxgen = 500):

    Path_short = Gena_TSP(data,self_num,matrix_distance,index,maxgen = maxgen)  # 根据位置坐标，生成一个遗传算法类

    Path_short.rand_chrom()  # 初始化父类

    ## 绘制初始化的路径图
    # draw(data,Path_short)

    # 循环迭代遗传过程
    for i in range(Path_short.maxgen):
        Path_short.select_sub()  # 选择子代
        Path_short.cross_sub()  # 交叉
        Path_short.mutation_sub()  # 变异
        Path_short.reverse_sub()  # 进化逆转
        Path_short.reins()  # 子代插入

        # 重新计算新群体的距离值
        for j in range(Path_short.size_pop):
            Path_short.fitness[j] = Path_short.comp_fit(Path_short.chrom[j, :])

        # 每隔三十步显示当前群体的最优路径
        index = Path_short.fitness.argmin()
        # print(index)
        if (i + 1) % 30 == 0:
            print('第' + str(i + 1) + '步后的最短的路程: ' + str(Path_short.fitness[index]))
            print('第' + str(i + 1) + '步后的最优路径:')
            # draw(data, Path_short)
            # print(Path_short.chrom)
            Path_short.out_path(Path_short.chrom[index, :])  # 显示每一步的最优路径

        # 存储每一步的最优路径及距离
        Path_short.best_fit.append(Path_short.fitness[index])
        Path_short.best_path.append(Path_short.chrom[index, :])
    return Path_short  # 返回遗传算法结果类



def matrix_dis(self_num,self_data):
    res = np.zeros((self_num,self_num))
    for i in range(self_num):
        for j in range(i+1,self_num):
            res[i,j] = np.linalg.norm(self_data[i,:]-self_data[j,:])
            res[j,i] = res[i,j]
    return res


# def test_out(he ):
#     he =  0
#     return  he
if __name__ == '__main__':
    data = map(25)[:]
    self_num = len(data)
    index = [i for i in range(self_num)]
    matrix_distance = matrix_dis(self_num,data)
    print(index,matrix_distance.shape)
    path = TSP(data,self_num,matrix_distance,index = index,maxgen = 500)



    # print(path)
    draw(data, path)
    res = path.out_put()
    for i in range(len(res)):

        path = TSP(data,res[i][0][:-1].shape[0],matrix_distance,index = res[i][0][:-1],maxgen = 90)
        draw(data, path)
    # print(path.out_put())
    # TSP.draw(data, path)
