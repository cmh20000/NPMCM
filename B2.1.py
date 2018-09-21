import numpy as np
import matplotlib.pyplot as plt
import xlrd as xr
import pickle as pk
import copy as cp
import pulp as pl

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

########################################################################################################################
# 一、数据结构：
#    维护点与点之间的距离，邻接矩阵
#    城市对象表：人口、城市名、边
#    设备流量-距离参照表
#    边表

# 1、图对象

# 节点
class city_info(object):
    def __init__(self, name, coor, p1, p2, w):
        self.name = name
        self.x = coor[0]
        self.y = coor[1]
        self.popu_city = p1
        self.popu_prov = p2
        self.weight = w
    def print(self):
        print(self.name, self.x, self.y, self.popu_city, self.popu_prov)
# 边
class connect(object):
    def __init__(self, city1, city2, volume):
        self.cid1 = city1
        self.cid2 = city2
        self.volume = volume
        self.weight = (city[self.cid1].weight + city[self.cid2].weight)/2
        self.popu_city_edge = np.sqrt(city[self.cid1].popu_city * city[self.cid2].popu_city)
        self.popu_prov_edge = np.sqrt(city[self.cid1].popu_prov * city[self.cid2].popu_prov)
        self.value_city = self.cal_value_city()
        self.value_prov = self.cal_value_prov()

    def cal_value_city(self):
        # 均值权重 * 容量 * 人口几何均值
        return self.weight * self.volume * self.popu_city_edge  #mTb/s
    def cal_value_prov(self):
        # 均值权重 * 容量 * 人口几何均值
        return self.weight * self.volume * self.popu_prov_edge  #mTb/s
    def refresh_data(self):
        self.weight = (city[self.cid1].weight + city[self.cid2].weight) / 2
        self.popu_city_edge = np.sqrt(city[self.cid1].popu_city * city[self.cid2].popu_city)
        self.popu_prov_edge = np.sqrt(city[self.cid1].popu_prov * city[self.cid2].popu_prov)
        self.value_city = self.cal_value_city()
        self.value_prov = self.cal_value_prov()

# 2、全局变量表
city = []
dis = []
dev = [(100, 3000, 8), (200, 1200, 16), (400, 600, 32)]  # 单量Gb/s、距离km、总容量Tb/s

edge_space = []
visit = []

best_edge_group = []
saver = {}
save_path = "save_2.1.tmp"

########################################################################################################################
# 二、基本数据辅助：
#    数据获取
#    计算路径总价值的函数
#    所有的备选路径解集生成

# 1、从excel读取数据
def build_data(filename):
    global dis
    file = xr.open_workbook(filename)
    table = file.sheets()[0]
    # 城市、坐标、市人口、省人口
    for i in range(table.nrows):
        tmp = table.row_values(i)
        city.append(city_info(tmp[0], (int(tmp[1]), int(tmp[2])), float(tmp[3]), float(tmp[4]), float(tmp[5])))
    # 距离
    table = file.sheets()[1]
    dis = np.zeros([table.nrows, table.ncols])
    for i in range(table.ncols):
        dis[:, i] = np.matrix(table.col_values(i))

# 2、路径总价值
def cal_whole_value(path, choose):
    # 默认权重为1 DZM
    sum = 0
    for i in range(len(path)):
        if choose == 0:
            sum += path[i].value_city
        else:
            sum += path[i].value_prov
    return sum

# 3、生成有效解集
def gen_solution_space():
    global visit
    # 所有在距离条件下的可行边，都建立起来
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            tmp = len(dev) - 1
            while True:
                if dis[i][j] <= dev[tmp][1]:
                    tmp_con = connect(i, j, dev[tmp][2])
                    edge_space.append(tmp_con)
                    break
                tmp -= 1
                if tmp < 0:
                    break


    # 城市间的可访问表
    visit = np.ones(dis.shape)*(-1)
    # visit = [[-1 for i in range(len(city))] for j in range(len(city))]
    for i in range(len(edge_space)):
        city_index1 = edge_space[i].cid1
        city_index2 = edge_space[i].cid2
        visit[city_index1][city_index2] = 0
        visit[city_index2][city_index1] = 0

########################################################################################################################
# 三、模拟退火框架
# 12个城市，求11个连线下，生成最大树，因为仅11个连线，故必然11个不同城市都有选择
# 状态量为11个选择，按城市顺序排，故前11个城市必可连接，
edge_table = []
last_index = -1

# 1、数据结构处理
# 1.1 构建edge_table
def build_edge_table():
    global edge_table
    edge_table = [[None for i in range(len(city))] for j in range(len(city))]
    for i in range(len(edge_space)):
        edge_table[edge_space[i].cid1][edge_space[i].cid2] = cp.deepcopy(edge_space[i])
        edge_table[edge_space[i].cid2][edge_space[i].cid1] = cp.deepcopy(edge_space[i])

def del_visit(x, y):
    visit[x][y] = 0
    visit[y][x] = 0

def gen_visit(x, y):
    visit[x][y] = 1
    visit[y][x] = 1

# 1.2 环检测
def loop_det(pos1, pos2):

    tmp_visit = [0 for i in range(len(city))]
    tmp_visit[pos1] = 1
    tmp_visit[pos2] = -1

    # 从左端pos1进行访问延伸，标记所有可访问到的点
    city_index = []
    for i in range(len(city)):
        # 本点周边除了pos2的连接点
        if visit[pos1][i] == 1 and i != pos2:
            city_index.append(i)
            tmp_visit[i] = 1

    while len(city_index) > 0:
        pos = city_index.pop()
        for i in range(len(city)):
            # 本点周边的连接点
            if visit[pos][i] == 1:
                if tmp_visit[i] == -1:
                    return True
                elif tmp_visit[i] == 0:
                    city_index.append(i)
                    tmp_visit[i] = 1

    # 从右端pos1进行访问延伸，标记所有可访问到的点
    return False

def loop_det_full(solution, pos1 = 0, pos2 = 0):
    for i in range(len(solution)):
        if loop_det(solution[i].cid1, solution[i].cid2):
            return True
    return False

# 1.3 同步当前解的结构到visit
def refresh_best_visit(best_solution):
    global last_index
    # print("before_refresh_visit_len: ", len(visit[visit == 1]))
    visit[visit == 1] = 0
    last_index = -1
    for i in range(len(best_solution)):
        gen_visit(best_solution[i].cid1, best_solution[i].cid2)
        if best_solution[i].cid1 == len(city)-1 or best_solution[i].cid2 == len(city)-1:
            last_index = i
    # print("after_refresh_visit_len: ", len(visit[visit == 1]))

# 1.4 数据结构调试
def compare_visit(solution):
    if len(visit[visit==1]) != 2*len(solution):
        print("length error", len(visit[visit==1]), 2*len(solution))
        return False
    for i in range(len(solution)):
        c1 = solution[i].cid1
        c2 = solution[i].cid2
        if visit[c1][c2] != 1:
            print("match error")
            return False
    return True

def compare_edge_tabel():
    for i in range(len(city)):
        for j in range(len(city)):
            if edge_table[i][j] == None:
                continue
            c1 = edge_table[i][j].cid1
            c2 = edge_table[i][j].cid2
            if (c1!=i and c2!=i) or (c1!=j and c2!=j):
                print("edge_tabel_error")
                return False
    return True

# 2、邻域探索函数，方案是改变后仍旧为树
# 2.1 完备性下的变更
def change_group(solution):
    global last_index
    pos = np.random.randint(0, len(city))  # 0-11
    # 1.1 最后一个
    if pos == 11:
        tmp_count = 20
        while True:
            tmp_count -= 1
            if tmp_count < 0:
                print("no adj available in ", pos)
                break
            # print("loop_change_group_12")
            target = np.random.randint(0, len(city) - 1)  # 0-10
            if visit[target][pos] == 0:
                # 删target原始的，填入target pos
                c1 = solution[target].cid1
                c2 = solution[target].cid2
                del_visit(c1, c2)
                gen_visit(pos, target)
                solution[target] = edge_table[pos][target]

                # 删last_index
                debug4(solution, 10.20)
                temp_count = 20
                while True:
                    temp_count -= 1
                    if temp_count < 0:
                        print("no adj available in ", last_index)
                        break
                    tmp_target = np.random.randint(0, len(city) - 1)  # 0-10
                    if visit[last_index][tmp_target] == 0:
                        del_visit(pos, last_index)
                        gen_visit(last_index, tmp_target)
                        solution[last_index] = edge_table[last_index][tmp_target]
                        break
                debug4(solution, 10.21)

                last_index = target
                debug3(solution, 0.1)
                break
    # 1.2 前十一个
    else:
        if pos == last_index:
            if pos + 1 > 10:
                pos -= 1
            elif pos - 1 < 0:
                pos += 1
            else:
                pos += [-1, 1][np.random.randint(0, 2)]

        tmp_count = 20
        while True:
            tmp_count -= 1
            if tmp_count < 0:
                print("no adj available in ", pos)
                break
            # print("loop_change_group")
            target = np.random.randint(0, len(city))  # 0-11
            if visit[pos][target] == 0:
                del_visit(solution[pos].cid1, solution[pos].cid2)
                gen_visit(pos, target)
                solution[pos] = edge_table[pos][target]
                debug3(solution, 0.2)
                break
    debug3(solution, 0.3)
    debug4(solution, 123)
    return solution

# 2.2 下一个路径组合的搜索策略
def find_next(edge_group, scale=2, min_num=2, max_num=8, pos=-1):
    global last_index
    global visit
    solution = edge_group
    # 还原点
    tmp_solution = cp.deepcopy(solution)
    count = scale
    debug4(solution, 100)  # 进入时

    # 1、随机挑一个，变更它的指向
    while True:
        # 防止无限循环
        count -= 1
        if count<0:
            # print("falied1,falied1,falied1")
            debug4(solution, 111)
            break
        if pos == -1:
            pos = np.random.randint(0, len(city))  # 0-11

        # 1.1 最后一个
        if pos == 11:
            debug3(solution, 12.1)
            debug4(solution, 12.1)  # 进入时
            tmp_count = 20
            while True:
                tmp_count -= 1
                if tmp_count < 0:
                    print("no adj available in ", pos)
                    break
                target = np.random.randint(0, len(city) - 1)  # 0-10

                if visit[target][pos] == 0:
                    # 删target原始的，填入target pos
                    loop_save = cp.deepcopy(solution[target])
                    c1 = solution[target].cid1
                    c2 = solution[target].cid2
                    del_visit(c1, c2)
                    gen_visit(pos, target)
                    solution[target] = edge_table[pos][target]

                    debug4(solution, 12.21)
                    # 删last_index
                    temp_count = 20
                    while True:
                        temp_count -= 1
                        if temp_count < 0:
                            print("no adj available in ", last_index)
                            break
                        tmp_target = np.random.randint(0, len(city) - 1)  # 0-10
                        if visit[last_index][tmp_target] == 0:
                            del_visit(pos, last_index)
                            gen_visit(last_index, tmp_target)
                            tmp_save = solution[last_index]
                            solution[last_index] = edge_table[last_index][tmp_target]
                            break
                    debug4(solution, 12.22)
                    break

            # debug3(solution, 12.2)
            debug4(solution, 12.2)  # 变形后

            # 环检测，完备性的必要条件
            if loop_det(pos, target):
                del_visit(pos, target)
                gen_visit(c1, c2)
                if temp_count >=0:
                    del_visit(last_index, tmp_target)
                    gen_visit(last_index, pos)
                    solution[last_index] = tmp_save
                solution[target] = cp.deepcopy(loop_save)
                debug4(solution, 12.3)  # 还原后
                continue
            else:
                last_index = target
                break

        # 1.2 前十一个
        else:
            if pos == last_index:
                if pos + 1 > 10:
                    pos -= 1
                elif pos - 1 < 0:
                    pos += 1
                else:
                    pos += [-1, 1][np.random.randint(0,2)]
            debug4(solution, 11.1)  # 进入时
            debug3(solution, 11.1)
            tmp_count = 20
            while True:
                tmp_count -= 1
                if tmp_count < 0:
                    print("no adj available in ", pos)
                    break
                target = np.random.randint(0, len(city))  # 0-11
                if visit[pos][target] == 0:
                    del_visit(solution[pos].cid1, solution[pos].cid2)
                    gen_visit(pos, target)
                    loop_save = cp.deepcopy(solution[pos])
                    solution[pos] = edge_table[pos][target]
                    break
            # debug3(solution, 11.2)
            debug4(solution, 11.2)  # 变形后

            # 环检测，完备性的必要条件
            if tmp_count >= 0:
                if loop_det(pos, target):
                    del_visit(pos, target)
                    gen_visit(loop_save.cid1, loop_save.cid2)
                    solution[pos] = cp.deepcopy(loop_save)
                    debug4(solution, 11.3)  # 还原后
                    continue
                else:
                    break

    # 2、第二套方案
    if count < 0:
        count = 100
        while True:
            count -= 1
            if count<10:
                print(count, end=' ')
            if count < 0:
                print("falied2,falied2,falied2,无限循环")

                #while True:
                #    print("falied2,falied2,falied2,无限循环")
                #    debug4(solution, 222)
                break
            # 2.1 变化多次
            tmp_num = np.random.randint(min_num, max_num)
            for i in range(2, tmp_num):
                solution = change_group(solution)
            debug3(solution, 22.1)
            debug4(solution, 22.1)  # 变形后

            if loop_det_full(solution):
                solution = cp.deepcopy(tmp_solution)
                refresh_best_visit(solution)
                debug4(solution, 22.2)  # 还原后
            else:
                break
    debug3(solution, 333)
    debug4(solution, 333)  # 离开时
    return solution

# 2.3 初始解的生成：读取或随机
def get_origin_solution(read_flag=True):
    global last_index
    global saver
    global visit
    solution = [None for i in range(len(city)-1)]
    last_index = -1
    if read_flag:
        try:
            with open(save_path, "rb") as fd:
                saver = pk.load(fd)
                solution = saver["solution"]
                for ele in solution:
                    ele.refresh_data()
                refresh_best_visit(solution)

                print("read file successfully")
                print("visit_len: ", len(visit[visit == 1]))
                debug4(solution, 1)
                return solution
        except:
            print("cannot find file " + save_path)
            pass

    while True:
        # 强制包含最后一个
        while True:
            pos = np.random.randint(0, len(city) - 1)
            if visit[pos][len(city) - 1] == 0:
                gen_visit(pos, len(city) - 1)
                solution[pos] = edge_table[pos][len(city) - 1]
                last_index = pos
                break
        # 仅前11个
        for i in range(len(city) - 1):
            if i == last_index:
                continue
            while True:
                target = np.random.randint(0, len(city))
                if visit[i][target] == 0:
                    gen_visit(i, target)
                    solution[i] = edge_table[i][target]
                    break

        # 环检测，完备性的必要条件
        if loop_det_full(solution):
            visit[visit == 1] = 0
            for i in range(len(solution)):
                solution[i] = None

            continue
        else:
            break
    debug3(solution, 111)
    return solution

# 2.4 支持手动构建，调试用
def change_by_hand(solution, name1, name2, index, loop_test_flag=True):
    c1 = -1
    c2 = -1
    for i in range(len(city)):
        if city[i].name == name1:
            c1 = i
        if city[i].name == name2:
            c2 = i
    if c1 == -1 or c2 == -1:
        print("Failed join " + name1 + ',' + name2)
        return -1
    tmp_save = cp.deepcopy(solution)
    solution[index] = edge_table[c1][c2]
    refresh_best_visit(solution)
    if loop_test_flag:
        if loop_det_full(solution):
            solution = cp.deepcopy(tmp_save)
            refresh_best_visit(solution)
            print("loop!")

    print("success!")

# 防止存档丢失，代码备份
def set_route_by_hand(solution):

    change_by_hand(solution, "北京&天津","武汉",0,False)
    change_by_hand(solution, "北京&天津", "哈尔滨", 1, False)
    change_by_hand(solution, "乌鲁木齐", "北京&天津", 2, False)
    change_by_hand(solution, "上海", "北京&天津", 3, False)
    change_by_hand(solution, "西安", "郑州", 4, False)
    change_by_hand(solution, "西安", "重庆", 5, False)
    change_by_hand(solution, "郑州", "武汉", 6, False)
    change_by_hand(solution, "重庆", "广州&深圳", 7, False)
    change_by_hand(solution, "成都", "重庆", 8, False)
    change_by_hand(solution, "拉萨", "北京&天津", 9, False)
    change_by_hand(solution, "昆明", "广州&深圳", 10, False)


# 3、模拟退火循环，主要处理前11个，后面的按序生成
# max_t变动幅度 k 越小越不容易接受
# 先find_scale大，k小，change_num小，尝试最近邻， 然后再反之

# 模拟退火变量：初始温度、末态温度、k值、温度衰减系数
(max_t, min_t, k, a) = (10000, 1, 0.25, 0.9)

# 邻域调整参数：
(scale, change_num_min, change_num_max, find_scale) = (3, 2, 10, 0.84)
# scale：尝试仅修改1条边的次数，如果scale次随机修改后仍旧不满足，那么启用多次修改
# change_num_min与change_num_max：多次修改时，修改次数是change_num_min到change_num_max之间的随机值
# find_scale：接受新值的补充概率，如果超过最优值的find_scale，也认为可以接受

choose = 0  # 0表示选择城市人口，1表示选择省（区）人口
total_num = 31  # 生成total_num 条连接

def SA():
    global saver
    print("Begin SA")
    build_edge_table()
    last_solution = get_origin_solution()

    #tmp = cp.deepcopy(last_solution)
    #set_route_by_hand(tmp)
    #debug5(tmp)

    best_solution = cp.deepcopy(last_solution)

    print("generate the initial solution")

    # 1、全局参数
    cur_t = max_t

    last_value = cal_whole_value(last_solution, choose)
    best_value = cal_whole_value(best_solution, choose)

    iter_count = 0
    iter_value = [best_value]

    pos = 0
    tmp = 0.1
    doc = np.log10(min_t/max_t)/np.log10(a)

    while cur_t > min_t:
        # 2、新的邻域解：
        new_solution = find_next(last_solution, scale, change_num_min, change_num_max, pos)
        pos += 1
        if pos > 11:
            pos = 0
        debug4(new_solution, 3)

        # 3、依概率接受
        new_value = cal_whole_value(new_solution, choose)
        if find_scale * best_value < new_value:
            last_solution = new_solution
            last_value = new_value
            #debug5(new_solution)
        else:
            r = np.random.uniform(0, 1)
            p = np.exp((new_value - best_value) / k / cur_t)
            if r < p:
                last_solution = new_solution
                last_value = new_value
                #debug5(new_solution)
            else:
                # visit 恢复为 last_solution
                refresh_best_visit(last_solution)

        # 4、周期导出
        if new_value > best_value:
            best_solution = cp.deepcopy(new_solution)
            best_value = new_value

        iter_count += 1
        cur_t *= a

        if iter_count > doc*tmp:
            print(tmp*100, "%")
            tmp += 0.1

        if iter_count % int(doc * 0.05) == 0:
            iter_value.append(last_value)

        if iter_count % int(doc * 0.33) == 0:
            saver["solution"] = best_solution
            with open(save_path, "wb") as fd:
                pk.dump(saver, fd)

    print("100%")
    print("iter_count: %d" % iter_count)
    print("iter_value_length: %d" % len(iter_value))
    print("best_value:", best_value)

    saver["solution"] = best_solution
    with open(save_path, "wb") as fd:
        pk.dump(saver, fd)

    # 价值变动显示
    plt.figure()
    plt.plot(range(len(iter_value)), iter_value)
    plt.show()

    # 一定注意visit是last_solution的，而不是best_solution的，一定要更新
    refresh_best_visit(best_solution)
    return best_solution

# 4、处理11个之后的连接
def build_remained_con(num):
    tmp_visit = np.zeros(dis.shape, dtype=np.int32)

    for i in range(len(best_edge_group)):
        c1 = best_edge_group[i].cid1
        c2 = best_edge_group[i].cid2
        tmp_visit[c1][c2] = 1
        tmp_visit[c2][c1] = 1

    remain_con = []
    for i in range(len(edge_space)):
        c1 = edge_space[i].cid1
        c2 = edge_space[i].cid2
        if tmp_visit[c1][c2] == 0:
            remain_con.append(edge_space[i])
            tmp_visit[c1][c2] = 1
            tmp_visit[c2][c1] = 1
    print("edge_space_len:", len(edge_space))
    print("remain_con_num: ", len(remain_con))
    print("best_edge_group_num: ", len(best_edge_group))
    print("visit_len: ", len(visit[visit == 1]))

    for i in range(len(best_edge_group)):
        gen_visit(best_edge_group[i].cid1, best_edge_group[i].cid2)

    if choose == 0:
        remain_con.sort(key=lambda ele: ele.value_city, reverse=True)
    else:
        remain_con.sort(key=lambda ele: ele.value_prov, reverse=True)
    for i in range(num-len(best_edge_group)):
        best_edge_group.append(remain_con[i])
        gen_visit(remain_con[i].cid1, remain_con[i].cid2)

    print("total_value:", cal_whole_value(best_edge_group, choose))

########################################################################################################################
# 四、最优中继系数（输入为网络连接图：visit + edge_table + city　输出优化结果）

# 目标值， k1 * abs(x1) + k2 * abs(x2)+..... 最大       去绝对值
# 边约束，要求 abs(x1)+....abx(xn) < edge_volume        去绝对值
# 节点的约束，要求 sum(adj_x) = 0

# 1、数据结构， 一个组合 = 1个流量X + num条边个分配比
# 1.1 AB 组合对应一个x
t_node = []
v_node = []
v_node_map = []  # 下标记录

# 1.2 一个边有len(v_node)个参数，因为可以做任意组合的中继，有正有负
t_edge = []
v_edge = []  # 有方向性，小坐标->大坐标为正，否则为负，表述方向
v_edge_map = []  # 下标记录

relay_con = []
best_lp_value = 0

def build_lp_data():
    # 命名：i,j 参数ID + 组合ID, 参数id=0的是流量，后面是所有边在此组合上的分配
    # 1、AB总容量
    global visit
    global t_node
    global v_node
    global v_node_map
    v_node_map = np.zeros(dis.shape, dtype=np.int32)
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            v_node.append(pl.LpVariable('X(%d,%d)' % (0, len(v_node)), lowBound=0))
            t_node.append(pl.LpVariable('T(%d,%d)' % (0, len(t_node)), lowBound=0))
            v_node_map[i][j] = len(v_node) - 1
            v_node_map[j][i] = len(v_node) - 1

    # 2、边num分配
    global v_edge_map
    global v_edge
    global t_edge
    v_edge_map = np.ones(dis.shape, dtype=np.int32) * np.int(-1)

    for i in range(len(city)):
        for j in range(i+1, len(city)):
            if visit[i][j] == 1:
                max_volume = edge_table[i][j].volume
                tmp_v_edge = [pl.LpVariable('X(%d,%d)' % (len(v_edge) + 1, k1), lowBound=-max_volume, upBound=max_volume)
                              for k1 in range(len(v_node))]
                tmp_t_edge = [pl.LpVariable('T(%d,%d)' % (len(t_edge) + 1, k2), lowBound=0, upBound=max_volume)
                              for k2 in range(len(t_node))]
                v_edge.append(tmp_v_edge)
                t_edge.append(tmp_t_edge)
                v_edge_map[i][j] = len(v_edge)-1
                v_edge_map[j][i] = len(v_edge)-1
    print(len(v_edge))

# 2、目标函数
def build_target():
    global choose
    target = 0
    # 去abs，取t最大
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            # 必须手写，因为可能是不存在的边
            if choose == 0:
                target += (city[i].weight+city[j].weight)/2 * np.sqrt(city[i].popu_city*city[j].popu_city) * v_node[v_node_map[i][j]]
            else:
                target += (city[i].weight+city[j].weight)/2 * np.sqrt(city[i].popu_prov*city[j].popu_prov) * v_node[v_node_map[i][j]]

    return target

# 3、约束方程
def get_adj_edge_v(center, v_node_id):
    adj = []
    # 一定注意符号问题，正：小标->大标 负：大标->小标
    for i in range(len(city)):
        if visit[center][i] == 1:
            if center > i:  # 说明正向是从center->i，可认为是出度
                adj.append(-1 * v_edge[v_edge_map[center][i]][v_node_id])
            else:   # 说明正向是从i->center，可认为是入度
                adj.append(v_edge[v_edge_map[center][i]][v_node_id])
    return adj

def build_constrain():
    cons = []
    # 3.1 点约束，流入流出的量相同
    # 比如是AB之间的传输
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            start = i
            end = j

            # 所有城市点约束
            error = 0
            for k in range(len(city)):
                value = 0
                if k == start:
                    value = v_node[v_node_map[i][j]]
                elif k == end:
                    value = -1 * v_node[v_node_map[i][j]]
                adj = get_adj_edge_v(k, v_node_map[i][j])
                cons.append(sum(adj) <= value + error)
                cons.append(sum(adj) >= value - error)

    # 3.2 边约束
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            if visit[i][j] == 1:
                value = edge_table[i][j].volume
                # 节点和约束
                cons.append(sum(t_edge[v_edge_map[i][j]]) <= value)
                # cons.append(sum(v_edge[v_edge_map[i][j]]) <= value)
                # 边abs约束
                for k in range(len(v_node)):
                    cons.append(v_edge[v_edge_map[i][j]][k] <= t_edge[v_edge_map[i][j]][k])
                    cons.append(-1 * v_edge[v_edge_map[i][j]][k] <= t_edge[v_edge_map[i][j]][k])

    # 3.3 流量绝对值约束
    for i in range(len(v_node)):
        cons.append(v_node[i] <= t_node[i])
        cons.append(-1 * v_node[i] <= t_node[i])

    return cons

# 4、优化框架
def lp_frame():
    global best_lp_value
    # 1、数据
    build_lp_data()
    target = build_target()
    cons = build_constrain()

    # 2、计算
    prob = pl.LpProblem('NPMCM', pl.LpMaximize)
    prob += target
    for con in cons:
        prob += con
    status = prob.solve()
    best_lp_value = pl.value(target)

    # 3、可视化

    # 3.1 各组合流量
    print("A-B城市间流量变动")
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            print("(", city[i].name, ',', city[j].name, "):", end=' ')
            # 原始没有边就是0，有边则是边值
            if visit[i][j] == 1:
                print("原始流量组合：", edge_table[i][j].volume, end=', ')
            else:
                print("原始流量组合：", 0, end=', ')
            print("现流量组合：", end=' ')
            # print(pl.value(v_node[v_node_map[i][j]]), end=',')
            print(pl.value(t_node[v_node_map[i][j]]))

    # 3.2 边流量分配
    '''
    print("边流量分配")
    for i in range(len(city)):
        for j in range(i+1, len(city)):
            if visit[i][j] == 1:
                print("连接 (", city[i].name, ',', city[j].name, "):")
                for m in range(len(city)):
                    for n in range(m + 1, len(city)):
                        print("     供应：(", city[m].name, ',', city[n].name, "):", end=' ')
                        print(pl.value(v_edge[v_edge_map[i][j]][v_node_map[m][n]]))
    '''
    # 3.3 打印中继
    threshold = 0.1
    find_flag = False
    for i in range(len(city)):
        for j in range(i + 1, len(city)):
            if visit[i][j] == 1:
                # 检查i,j线是否为中继线
                find_flag = False
                for m in range(len(city)):
                    for n in range(m + 1, len(city)):
                        # 本线的流量变动了
                        if m == i and n ==j:
                            if abs(pl.value(v_edge[v_edge_map[i][j]][v_node_map[m][n]])) <= edge_table[i][j].volume - threshold:
                                if not find_flag:
                                    print("中继线路 (", city[i].name, ',', city[j].name, "):")
                                    find_flag = True
                                print("     牺牲：(", city[m].name, '->', city[n].name, "):", end=' ')
                                print(edge_table[i][j].volume - pl.value(v_edge[v_edge_map[i][j]][v_node_map[m][n]]),"mTb/s")
                            continue
                        # 运载的流量目标
                        if abs(pl.value(v_edge[v_edge_map[i][j]][v_node_map[m][n]])) >= threshold:
                            if not find_flag:
                                print("中继线路 (", city[i].name, ',', city[j].name, "):")
                                find_flag = True
                            print("     供应：(", city[m].name, '->', city[n].name, "):", end=' ')
                            print(pl.value(v_edge[v_edge_map[i][j]][v_node_map[m][n]]), "mTb/s")
                            relay_con.append((i, j, m, n))

    print("原价值：", cal_whole_value(best_edge_group, choose))
    print("现价值：", best_lp_value)

########################################################################################################################
# 五、可视化：
# 1、城市点分布，根据人口划分大小，如果是中继点画蓝色
def draw_citys(dis_flag= False):
    for i in range(len(city)):
        plt.plot(city[i].x, city[i].y, marker='o', markerfacecolor='red', color='black')
        plt.text(city[i].x, city[i].y+1, city[i].name, ha='center', va='bottom', fontsize=10)
    if dis_flag:
        for i in range(len(city)):
            for j in range(i + 1, len(city)):
                plt.plot([city[i].x, city[j].x], [city[i].y, city[j].y], color='blue', linestyle='--', linewidth=0.1)
                plt.text((city[i].x + city[j].x) / 2, (city[i].y + city[j].y) / 2 + 1, str(dis[i][j]) + 'km', ha='center', va='bottom', fontsize=7)
    pass

# 2、结果线说明，颜色区别流量
def draw_result(solution=best_edge_group):
    c = []
    cmap = ["red", "blue", "green"]
    label = ["100G/s", "200G/s", "400G/s"]
    for i in range(len(solution)):
        tmp = len(dev) - 1
        tmp_con = solution[i]
        while True:
            if tmp_con.volume == dev[tmp][2]:
                c.append(cmap[tmp])
                break
            tmp -= 1
            if tmp < 0:
                print(tmp_con.volume)

        city_index1 = tmp_con.cid1
        city_index2 = tmp_con.cid2
        if i < len(city) - 1:
            plt.plot([city[city_index1].x, city[city_index2].x], [city[city_index1].y, city[city_index2].y], color=c[i],
                     linestyle='-', linewidth=1.2)
        else:
            plt.plot([city[city_index1].x, city[city_index2].x], [city[city_index1].y, city[city_index2].y], color=c[i],
                     linestyle='-', linewidth=0.7)
        plt.text((city[city_index1].x + city[city_index2].x) / 2,
                 (city[city_index1].y + city[city_index2].y) / 2 + 1,
                 str(i), ha='center', va='bottom', fontsize=7)
    # str(tmp_con.value)
    # '''
    p1, = plt.plot([60, 60], [60, 60], color="red", linestyle='-', linewidth=2)
    p2, = plt.plot([60, 60], [60, 60], color="blue", linestyle='-', linewidth=2)
    p3, = plt.plot([60, 60], [60, 60], color="green", linestyle='-', linewidth=2)

    plt.legend([p1,p2,p3], label, loc='upper center')
    # '''
    pass

# 3、中继分配线与对应说明
def draw_relay_con():
    tmp_visit = [0 for i in range(len(city))]
    c = "blue"
    cmap = ["blue","yellow", "green"]
    label = ["中继点", "收益点", "既是中继又是收益点", "中继路径"]
    p = [None, None, None, None]

    for t1, t2, c1, c2 in relay_con:
        for i in [t1,t2]:
            # 剥削者
            if i in [c1, c2]:
                if tmp_visit[i] == -1:
                    c = "green"
                    tmp_visit[i] = 9
                elif tmp_visit[i] == 0:
                    c = "yellow"
                    tmp_visit[i] = 1
                else:
                    c = "none"
            # 劳动者
            else:
                if tmp_visit[i] == 1:
                    c = "green"
                    tmp_visit[i] = 9
                elif tmp_visit[i] == 0:
                    c = "blue"
                    tmp_visit[i] = -1
                else:
                    c = "none"

            if c!="none":
                plt.plot(city[i].x, city[i].y, marker='o', markerfacecolor=c, color='black')

        plt.plot([city[t1].x, city[t2].x], [city[t1].y, city[t2].y], color= "black", linestyle= '-', linewidth= 1.2)
        '''
        plt.text((city[t1].x + city[t2].x) / 2,
                 (city[t1].y + city[t2].y) / 2 + 1,
                 city[c1].name + ',' + city[c2].name, ha='center', va='bottom', fontsize=7)
        '''

        for i in [c1, c2]:
            # 剥削者
            if tmp_visit[i] == -1:
                c = "green"
                tmp_visit[i] = 9
            elif tmp_visit[i] == 0:
                c = "yellow"
                tmp_visit[i] = 1
            else:
                c = "none"
            if c != "none":
                plt.plot(city[i].x, city[i].y, marker='o', markerfacecolor=c, color='black')
    # '''
    tmp_x = 44
    tmp_y = 90
    p[0], = plt.plot(tmp_x, tmp_y, marker='o', markerfacecolor="blue", color='black')
    p[1], = plt.plot(tmp_x, tmp_y, marker='o', markerfacecolor="yellow", color='black')
    p[2], = plt.plot(tmp_x, tmp_y, marker='o', markerfacecolor="green", color='black')
    p[3], = plt.plot([tmp_x, tmp_x], [tmp_y, tmp_y], color= "black", linestyle= '-', linewidth= 1.2)
    plt.legend(p, label, loc="upper center")
    # '''

########################################################################################################################
# 六、调试
def debug1():
    for i in range(len(city)):
        city[i].print()
    print(dis)

def debug2(num):
    plt.figure("result")
    plt.subplot("121")
    draw_citys()
    draw_result(best_edge_group)
    plt.title("未分配的最优总价值为：" + str(cal_whole_value(best_edge_group, choose)) + "(mTb/s)")
    plt.xlabel("边数:" + str(num))

    plt.subplot("122")
    draw_citys()
    draw_relay_con()
    plt.title("分配后的最优总价值为：" + str(best_lp_value) + "(mTb/s)")
    plt.xlabel("边数:" + str(num))

    plt.show()

def debug3(solution, pos = -1):
    for i in range(len(solution)):
        if solution[i].cid1 != i and solution[i].cid2 != i:
            print("solution index error in ", i)
            while True:
                pass
    if solution[last_index].cid1 != 11 and solution[last_index].cid2 != 11:
        print("last_index error in ", last_index, " pos:", pos)
        while True:
            pass

    pass

def debug4(solution, pos = -1):
    if not compare_visit(solution):
        while True:
            print("visit_error", len(visit[visit == 1]), len(solution), pos)
            exit(0)
    if not compare_edge_tabel():
        while True:
            compare_edge_tabel()
            exit(0)
def debug5(solution):
    plt.figure("tmp")
    draw_citys()
    draw_result(solution)
    plt.title("未分配11边的最优总价值为：" + str(cal_whole_value(solution, choose)))
    plt.show()


def opt_frame(num):
    global best_edge_group
    build_data("./city.xls")
    gen_solution_space()

    best_edge_group = SA()
    build_remained_con(num)
    lp_frame()

    debug2(num)
    pass

if __name__ == "__main__":
    opt_frame(total_num)

