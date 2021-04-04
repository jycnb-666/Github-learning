"""coding = utf-8"""
"""最优潮流主程序"""

from IEEE_5 import ieee_5   # 导入网络结构
from createY import createY    # 节点导纳矩阵形成函数
from coeff import  coeff    # 雅克比矩阵形成函数
from d_x import d_x     # 修正量求取函数
from diag import make_diag     # 将1-D矩阵构造为对角矩阵
from diag import get_diag   # 提取主对角元函数
from list_tran import list_tran  # 列表转置
from mat_out import mat_out # 格式化输出矩阵
import numpy as np
import time     # 程序计时

time_start = time.time()    # 程序开始时间

[N, B, G] = ieee_5()
(num_node, t1) = np.shape(N)   # 系统节点个数
(num_branch, t2) = np.shape(B)  # 系统支路个数

# 计算系统PQ、PV类型节点个数
node_type = N[:,2]
num_PQ = np.sum(node_type == 0)
num_PV = np.sum(node_type == 2)

num_gen = num_node - num_PQ     # 发电机个数
num_equa = 2 * num_node     # 等式约束个数m
num_inequa = 2 * num_gen + num_node + num_branch     # 不等式约束个数r
num_variable = 2 * num_gen + 2 * num_node     # 总变量个数n
P_load = N[1:num_PQ, 5]
Q_load = N[1:num_PQ, 6]

# 耗量特性多项式系数
a_0 = G[:,8]
a_1 = G[:,7]
a_2 = G[:,6]
A_1 = np.diag(a_1)
A_2 = np.diag(a_2)

# 形成节点导纳矩阵
Y = createY(N, B, num_node, num_branch)

# 发电机组有功出力和无功出力
P_G = np.zeros([num_gen,1])
Q_R = np.zeros([num_gen,1])
P_G[:,0] = N[num_PQ:num_node,7]
Q_R[:,0] = N[num_PQ:num_node,8]
# 节点功率注入
P_ejec = N[:,7]-N[:,5]
Q_ejec = N[:,8]-N[:,6]

x_tilde = np.zeros([2*num_node,1])      # 状态变量
x_tilde[[x for x in range(0,num_node*2) if x % 2 == 0],0] = N[:,4]    # 相角
x_tilde[[x for x in range(0,num_node*2) if x % 2 == 1],0] = N[:,3]    # 电压
u = np.ones([num_inequa,1])     # 松弛变量
l = np.ones([num_inequa,1])
z = np.ones([1,num_inequa])     # 拉格朗日乘子
w = -0.5 * np.ones([1,num_inequa])
y = np.ones(num_equa)
y[[x for x in range(0,num_equa) if x % 2 == 0]] = (1e-10) * np.ones(int(num_equa/2))
y[[x for x in range(0,num_equa) if x % 2 == 1]] = -(1e-10) * np.ones(int(num_equa/2))
y = y.reshape([1,num_equa])

# 迭代参数
epsi = 1e-6     # 收敛条件
sigma = 0.1     # 中心参数
max_iteration = 1      # 最大迭代次数
gap_record = []    # 记录对偶间隙Gap的迭代过程

g_u = G[:,2].tolist() + G[:,3].tolist() + N[:,17].tolist() + B[:,11].tolist()   # 不等式约束上界
g_u = np.array(g_u).reshape([len(g_u),1])
g_l = G[:,4].tolist() + G[:,5].tolist() + N[:,18].tolist() + (-B[:,11]).tolist()  # 不等式约束下界
g_l = np.array(g_l).reshape([len(g_l),1])

# 优化迭代主程序
for num_iteration in range(1,max_iteration+1):
    gap = (np.matmul(l.T,z.T) - np.matmul(u.T,w.T))[0][0]
    gap_record.append(gap)
    if gap < epsi:
        print("迭代结束！")
        break
    mu = gap * sigma/(2*num_inequa)     # 扰动因子
    x_variable = P_G.tolist() + Q_R.tolist() + x_tilde.tolist()    # 总变量
    x_fix = z.T.tolist() + l.tolist() + w.T.tolist() +\
            u.tolist() + x_variable + y.T.tolist()    # 修正量


    # 构建不等式约束
    g_1 = P_G       # 有功出力
    g_2 = Q_R       # 无功出力
    g_3 = np.zeros([num_node,1])
    g_3 = x_tilde[[x for x in range(0,num_node*2) if x % 2 == 1],:]    # 节点电压幅值
    g_4 = np.zeros([num_branch,1])# 线路潮流
    for k in range(0,num_branch):
        branch_i = int(B[k,0])-1      # 支路始节点
        branch_j = int(B[k,1])-1      # 支路末节点
        theta = x_tilde[(branch_i-1)*2] - x_tilde[(branch_j-1)*2]     # 相角差
        g_4[k,0] = x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                 (Y[branch_i,branch_j].real * np.cos(theta) +
                  Y[branch_i,branch_j].imag * np.sin(theta)) -\
                 x_tilde[2*branch_i+1]**2 * Y[branch_i,branch_j].real
    g = g_1.tolist() + g_2.tolist() + g_3.tolist() + g_4.tolist()    # 不等式约束

    # 构建等式约束
    h = np.zeros([num_node*2,1])
    for i in range(0,num_node):
        for j in range(0,num_node):
            theta = x_tilde[i*2] - x_tilde[j*2]
            h[i*2,:] = h[i*2,:] - x_tilde[i*2+1] * x_tilde[j*2+1] * \
                     (Y[i,j].real * np.cos(theta) + Y[i,j].imag * np.sin(theta))
            h[i*2+1,:] = h[i*2+1,:] - x_tilde[i*2+1] * x_tilde[j*2+1] * \
                     (Y[i,j].real * np.sin(theta) - Y[i,j].imag * np.cos(theta))
        h[i*2,:] = h[i*2,:] + P_ejec[i]
        h[i*2+1,:] = h[i*2+1,:] + Q_ejec[i]

    # 形成拉格朗日函数
    L = make_diag(l)
    U = make_diag(u)
    Z = make_diag(z)
    W = make_diag(w)
    # 拉格朗日函数偏导数
    L_y_ = h
    L_z_ = g - l - g_l
    L_w_ = g + u - g_u
    L_mu_l = np.matmul(np.matmul(L,Z), np.ones([num_inequa,1])) - mu * np.ones([num_inequa,1])
    L_mu_u = np.matmul(np.matmul(U,W), np.ones([num_inequa,1])) + mu * np.ones([num_inequa,1])

    # Jacobian矩阵的形成
    [A, dh_dx, dg_dx, H_, d2h_dx_y, d2g_dx_c, d2f_dx, temp, L_Z, U_W] = coeff(num_node, num_PQ, x_tilde,
                                                                              Y, B, x_fix, A_2,num_iteration)
    # 求目标函数梯度矢量
    A_1_diag = get_diag(A_1)
    df_dx = np.concatenate((2 * np.matmul(A_2, P_G) + A_1_diag, np.zeros([2,1]), np.zeros([len(x_tilde),1])), axis=0)
    # 拉格朗日函数对x偏导数
    L_x = df_dx - np.matmul(dh_dx, y.T) - np.matmul(dg_dx, (z+w).T)
    L_x_ = L_x + np.matmul(dg_dx,(np.matmul(np.linalg.inv(L),(L_mu_l + np.matmul(Z, L_z_))) +
                                  np.matmul(np.linalg.inv(U),(L_mu_u - np.matmul(W, L_w_)))))

    b = np.concatenate((-np.matmul(np.linalg.inv(L),L_mu_l), L_z_,
                        -np.matmul(np.linalg.inv(U),L_mu_u), -L_w_, L_x_, -L_y_), axis=0)

    delta_x = d_x(H_, dg_dx, dh_dx, L_Z, U_W, A, b, z, l, u, w, x_variable, y)

    # 更新修正量
    x_fix = x_fix + delta_x
    z = (x_fix[0:num_inequa]).T
    l = x_fix[num_inequa:2*num_inequa]
    w = (x_fix[2*num_inequa:3*num_inequa]).T
    u = x_fix[3*num_inequa:4*num_inequa]
    x = x_fix[4*num_inequa:5*num_inequa]
    y = (x_fix[5*num_inequa:5*num_inequa+num_equa]).T

    P_G = x[0:num_gen]
    Q_R = x[num_gen:2*num_gen]
    P_ejec[num_PQ:num_node] = P_G[:,0]
    Q_ejec[num_PQ:num_node] = Q_R[:,0]
    X_tilde = x[2*num_gen:2*num_gen+2*num_node]

    print("第" + str(num_iteration) +  "次迭代的对偶间隙Gap为：{0.real:8.2f}".format(gap))
time_end = time.time() # 程序结束时间
print("本次运行共耗时{0.real:.3f}毫秒".format((time_end-time_start)*1000))