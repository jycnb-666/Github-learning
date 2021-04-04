"""coding = utf8"""
import numpy as np
import pandas as pd
from mat_out import mat_out

def coeff(num_node, num_PQ, x_tilde, Y, B, X, A2, num_iteration):
    """形成最优潮流迭代过程中的雅克比矩阵"""

    # 确定各指标数量信息
    (num_branch,t) = np.shape(B)        # 支路个数
    len_x = 2 * (num_node - num_PQ + num_node)      # 总变量（控制变量+状态变量）个数
    num_gen = num_node - num_PQ     # 发电机个数
    num_equa = 2 * num_node
    num_inequa = 2 * num_gen + num_node + num_branch


    # 确定当前修正量
    z = np.array(X[0 : num_inequa])
    l = np.array(X[num_inequa : 2*num_inequa])
    w = np.array(X[2*num_inequa : 3*num_inequa])
    u = np.array(X[3*num_inequa : 4*num_inequa])
    x = np.array(X[4*num_inequa : 4*num_inequa+len_x])
    y = np.array(X[4*num_inequa+len_x : 4*num_inequa+len_x+num_equa])

    # 一、形成系数矩阵
    # 第一步 计算等式约束的雅克比矩阵
    dh_dP_G = np.zeros([2, 2*num_node])
    dh_dQ_R = np.zeros([2, 2*num_node])
    dh_dx_tilde = np.zeros([2*num_node, 2*num_node])    # 潮流计算的雅克比矩阵
    for k in range(num_PQ, num_node):
        dh_dP_G[k-num_PQ, k*2] = 1
        dh_dQ_R[k-num_PQ, k*2+1] = 1
    for i in range(0,num_node):
        for j in range(0,num_node):
            if i!=j:
                theta = x_tilde[2*i] - x_tilde[2*j]
                # 非对角元
                dh_dx_tilde[(2*j, 2*i)] = -x_tilde[2*i+1] * x_tilde[2*j+1] * \
                                          (Y[i,j].real * np.sin(theta) - Y[i,j].imag * np.cos(theta))
                dh_dx_tilde[(2*j, 2*i+1)] = x_tilde[2*i+1] * x_tilde[2*j+1] * \
                                            (Y[i, j].real * np.cos(theta) + Y[i, j].imag * np.sin(theta))
                dh_dx_tilde[(2*j+1, 2*i)] = -x_tilde[2*i+1] * \
                                            (Y[i,j].real * np.cos(theta) + Y[i,j].imag * np.sin(theta))
                dh_dx_tilde[(2*j+1, 2*i+1)] = -x_tilde[2*i+1] * \
                                            (Y[i,j].real * np.sin(theta) - Y[i,j].imag * np.cos(theta))
                # 主对角元
                dh_dx_tilde[(2*i, 2*i)] = dh_dx_tilde[(2*i, 2*i)] + x_tilde[2*i+1] * x_tilde[2*j+1] * \
                                          (Y[i,j].real * np.sin(theta) - Y[i,j].imag * np.cos(theta))
                dh_dx_tilde[(2*i, 2*i+1)] = dh_dx_tilde[(2*i, 2*i+1)] - x_tilde[2*i+1] * x_tilde[2*j+1] * \
                                            (Y[i, j].real * np.cos(theta) + Y[i, j].imag * np.sin(theta))
                dh_dx_tilde[(2*i+1, 2*i)] = dh_dx_tilde[(2*i+1, 2*i)] - x_tilde[2*j+1] * \
                                            (Y[i,j].real * np.cos(theta) + Y[i,j].imag * np.sin(theta))
                dh_dx_tilde[(2*i+1, 2*i+1)] = dh_dx_tilde[(2*i+1, 2*i+1)] - x_tilde[2*j+1] * \
                                              (Y[i,j].real * np.sin(theta) - Y[i,j].imag * np.cos(theta))

        dh_dx_tilde[(2*i+1, 2*i)] = dh_dx_tilde[(2*i+1, 2*i)] - 2 * x_tilde[2*i+1] * Y[i,i].real
        dh_dx_tilde[(2*i+1, 2*i+1)] = dh_dx_tilde[(2*i+1, 2*i+1)] + 2 * x_tilde[2*i+1] * Y[i,i].imag
    dh_dx = np.concatenate((dh_dP_G,dh_dQ_R, dh_dx_tilde), axis=0)
    # 输出计算等式约束的雅克比矩阵
    # print("第" + str(num_iteration) + "次迭代的等式约束雅克比矩阵为：")
    # mat_out(dh_dx)
    del dh_dP_G,dh_dQ_R,dh_dx_tilde

    # 第二步 计算不等式约束的雅克比矩阵
    dg1_dP_G = np.eye(num_gen)
    dg1_dQ_R = np.zeros([num_gen, num_gen])
    dg1_dx_tilde = np.zeros([2*num_node, num_gen])
    dg1_dx = np.concatenate((dg1_dP_G, dg1_dQ_R.tolist(), dg1_dx_tilde),axis=0)

    dg2_dP_G = np.zeros([num_gen, num_gen])
    dg2_dQ_R = np.eye(num_gen)
    dg2_dx_tilde = np.zeros([2*num_node, num_gen])
    dg2_dx = np.concatenate((dg2_dP_G, dg2_dQ_R, dg2_dx_tilde),axis=0)

    dg3_dP_G = np.zeros([num_gen, num_node])
    dg3_dQ_R = np.zeros([num_gen, num_node])
    dg3_dx_tilde = np.zeros([2*num_node, num_branch])
    for i in range(0,num_node):
        dg3_dx_tilde[2*i+1,i] = 1
    dg3_dx = np.concatenate((dg3_dP_G, dg3_dQ_R, dg3_dx_tilde),axis=0)

    dg4_dP_G = np.zeros([num_gen, num_node])
    dg4_dQ_R = np.zeros([num_gen, num_node])
    dg4_dx_tilde = np.zeros([2*num_node, num_branch])
    for k in range(0,num_branch):
        branch_i = int(B[k,0])-1
        branch_j = int(B[k,1])-1
        theta = x_tilde[2*branch_i] - x_tilde[2*branch_j]
        dg4_dx_tilde[2*branch_i, k] = -x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                                      (Y[branch_i,branch_j].real * np.sin(theta) -
                                       Y[branch_i,branch_j].imag * np.cos(theta))
        dg4_dx_tilde[2*branch_j, k] = x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                                      (Y[branch_i,branch_j].real * np.sin(theta) -
                                       Y[branch_i,branch_j].imag * np.cos(theta))
        dg4_dx_tilde[2*branch_i+1, k] = x_tilde[2*branch_j+1] * \
                                        (Y[branch_i,branch_j].real * np.cos(theta) +
                                         Y[branch_i,branch_j].imag * np.sin(theta)) - \
                                        2 * x_tilde[2*branch_i+1] * Y[branch_i,branch_j].real
        dg4_dx_tilde[2*branch_j+1, k] = x_tilde[2*branch_i+1] * \
                                        (Y[branch_i,branch_j].real * np.cos(theta) +
                                         Y[branch_i,branch_j].imag * np.sin(theta))
    dg4_dx = np.concatenate((dg4_dP_G, dg4_dQ_R, dg4_dx_tilde),axis=0)

    dg_dx = np.concatenate((dg1_dx, dg2_dx, dg3_dx, dg4_dx), axis=1)
    # 输出计算不等式约束的雅克比矩阵
    # print("第" + str(num_iteration) + "次迭代的不等式约束雅克比矩阵为：")
    # mat_out(dg_dx)
    del dg1_dP_G, dg1_dQ_R, dg1_dx_tilde, dg1_dx, \
        dg2_dP_G, dg2_dQ_R, dg2_dx_tilde, dg2_dx, \
        dg3_dP_G, dg3_dQ_R, dg3_dx_tilde, dg3_dx, \
        dg4_dP_G, dg4_dQ_R, dg4_dx_tilde, dg4_dx

    # 第三步 计算对角矩阵
    L_Z = np.eye(num_inequa) * np.divide(z,l)
    U_W = np.eye(num_inequa) * np.divide(w,u)
    # 输出计算出的对角矩阵
    # print("第" + str(num_iteration) + "次迭代的对角矩阵分别为：")
    # mat_out(L_Z)
    # mat_out(U_W)

    # 第四步 计算Hessian矩阵
    # 目标函数的Hessian矩阵
    d2f_dx = np.zeros([len_x,len_x])
    d2f_dx[0:num_gen,0:num_gen] = 2 * A2

    # 计算等式约束的Hessian矩阵与Lagrange乘子y乘积
    d2h_dx_y = np.zeros([len_x, len_x])
    A = np.zeros([2*num_node,2*num_node])
    for i in range(0,num_node):
        for j in range(0,num_node):
            theta = x_tilde[2*i] - x_tilde[2*j]
            if i!=j:
                # 需累加
                A[2*i, 2*i] = A[2*i, 2*i] + x_tilde[2*i+1] * x_tilde[2*j+1] * (Y[i,j].real * \
                              (np.cos(theta) * y[2*i] + np.sin(theta) * y[2*i+1] + \
                               np.cos(theta) * y[2*j] - np.sin(theta) * y[2*j+1]) + \
                              Y[i,j].imag * (np.sin(theta) * y[2*i] - np.cos(theta) * y[2*i+1] - \
                              np.sin(theta) * y[2*j] - np.cos(theta) * y[2*j+1]))
                A[2*i, 2*i+1] = A[2*i, 2*i+1] + x_tilde[2*j+1] * (Y[i,j].real * \
                                (np.sin(theta) * y[2*i] - np.cos(theta) * y[2*i+1] + \
                                 np.sin(theta) * y[2*j] + np.cos(theta) * y[2*j+1])+ \
                                Y[i,j].imag * (-np.cos(theta) * y[2*i] - np.sin(theta) * y[2*i+1] + \
                                np.cos(theta) * y[2*j] - np.sin(theta) * y[2*j+1]))
                A[2*i+1, 2*i] = A[2*i+1, 2*i] + x_tilde[2*j+1] * (Y[i,j].real *
                                (np.sin(theta) * y[2*i] - np.cos(theta) * y[2*i+1] + \
                                 np.sin(theta) * y[2*j] + np.cos(theta) * y[2*j+1])+ \
                                Y[i,j].imag * (-np.cos(theta) * y[2*i] - np.sin(theta) * y[2*i+1] + \
                                np.cos(theta) * y[2*j] - np.sin(theta) * y[2*j+1]))
                # 不需累加
                A[2*i, 2*j] = x_tilde[2*i+1] * x_tilde[2*j+1] * (Y[i,j].real * \
                              (-np.cos(theta) * y[2*i] - np.sin(theta) * y[2*i+1] - \
                               np.cos(theta) * y[2*j] + np.sin(theta) * y[2*j+1]) + \
                              Y[i,j].imag * (-np.sin(theta) * y[2*i] + np.cos(theta) * y[2*i+1] + \
                              np.sin(theta) * y[2*j] + np.cos(theta) * y[2*j+1]))
                A[2*i, 2*j+1] = x_tilde[2*i+1]*(Y[i, j].real * \
                               (np.sin(theta) * y[2*i] - np.cos(theta) * y[2*i+1] + \
                                np.sin(theta) * y[2*j] + np.cos(theta) * y[2*j+1]) + \
                               Y[i,j].imag * (-np.cos(theta) * y[2*i] - np.sin(theta) * y[2*i+1] + \
                               np.cos(theta) * y[2*j] - np.sin(theta) * y[2*j+1]))
                A[2*i+1, 2*j] = x_tilde[2*j+1] * (Y[i,j].real * \
                                (-np.sin(theta) * y[2*i] + np.cos(theta) * y[2*i+1] - \
                                 np.sin(theta) * y[2*j] - np.cos(theta) * y[2*j+1]) + \
                                Y[i, j].imag * (np.cos(theta) * y[2*i] + np.sin(theta) * y[2*i+1] -
                                np.cos(theta) * y[2*j] + np.sin(theta) * y[2*j+1]))
                A[2*i+1, 2*j+1] = -(Y[i,j].real * (np.cos(theta) * y[2*i] + np.sin(theta) * y[2*i+1] + \
                                                    np.cos(theta) * y[2*j] - np.sin(theta) * y[2*j+1]) + \
                                    Y[i,j].imag * (np.sin(theta) * y[2*i] - np.cos(theta) * y[2*i+1] - \
                                                    np.sin(theta) * y[2*j] - np.cos(theta) * y[2*j+1]))
        A[2*i+1, 2*i+1] = -2 * (Y[i,i].real * y[2*i] - Y[i,i].imag * y[2*i+1])
    d2h_dx_y[2*num_gen:len_x, 2*num_gen:len_x] = A
    # print("第" + str(num_iteration) + "次迭代的等式约束的Hessian矩阵与Lagrange乘子y乘积为：")
    # mat_out(d2h_dx_y)
    del A

    # 计算不等式约束Hessian矩阵与Lagrange乘子c=z+w的乘积
    d2_g_dx_c = np.zeros([len_x,len_x])
    d2_g4_d2_x_title = np.zeros([len(x_tilde),len(x_tilde)])
    c = z + w

    for k in range(0,num_branch):
        branch_i = int(B[k,0]) - 1
        branch_j = int(B[k,1]) - 1
        theta = x_tilde[2*(branch_i-1)] - x_tilde[2*(branch_j-1)]

        d2_g4_d2_x_title[2*branch_i, 2*branch_i] = d2_g4_d2_x_title[2*branch_i, 2*branch_i] + \
                                                   (-x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                                                    (Y[branch_i,branch_j].real * np.cos(theta) + Y[branch_i,branch_j].imag * np.sin(theta))) * \
                                                   c[2+2+5+k-1]

        d2_g4_d2_x_title[2*branch_i, 2*branch_i+1] = d2_g4_d2_x_title[2*branch_i, 2*branch_i+1] + (-x_tilde[2*branch_j+1] * \
                                                    (Y[branch_i,branch_j].real * np.sin(theta) - Y[branch_i,branch_j].imag * np.cos(theta))) * \
                                                     c[2+2+5+k-1]

        d2_g4_d2_x_title[2*branch_i, 2*branch_j] = d2_g4_d2_x_title[2*branch_i, 2*branch_j] + \
                                                   (x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                                                    (Y[branch_i,branch_j].real * np.cos(theta) + Y[branch_i,branch_j].imag * np.sin(theta))) * \
                                                    c[2+2+5+k-1]

        d2_g4_d2_x_title[2*branch_i, 2*branch_j+1] = d2_g4_d2_x_title[2*branch_i, 2*branch_j+1] + \
                                                     (-x_tilde[2*branch_i+1] * (Y[branch_i,branch_j].real * np.sin(theta) - Y[branch_i,branch_j].imag * np.cos(theta))) * \
                                                     c[2+2+5+k-1]
        d2_g4_d2_x_title[2*branch_j, 2*branch_i] = d2_g4_d2_x_title[2*branch_i, 2*branch_j]
        d2_g4_d2_x_title[2*branch_j, 2*branch_i+1] = d2_g4_d2_x_title[2*branch_j, 2*branch_i+1] + (x_tilde[2*branch_j+1] * \
                                                    (Y[branch_i,branch_j].real * np.sin(theta) - Y[branch_i,branch_j].imag * np.cos(theta))) * \
                                                     c[2+2+5+k-1]
        # d2_g4_d2_x_title[2*branch_j, 2*branch_j] = d2_g4_d2_x_title[2*branch_i, 2*branch_i]
        d2_g4_d2_x_title[2*branch_j, 2*branch_j] = d2_g4_d2_x_title[2*branch_j, 2*branch_j] + \
                                                   (-x_tilde[2*branch_i+1] * x_tilde[2*branch_j+1] * \
                                                    (Y[branch_i,branch_j].real * np.cos(theta) + Y[branch_i,branch_j].imag * np.sin(theta))) * \
                                                   c[2+2+5+k-1]
        d2_g4_d2_x_title[2*branch_j, 2*branch_j+1] = d2_g4_d2_x_title[2*branch_j, 2*branch_j+1] + (x_tilde[2*branch_i+1] * \
                                                    (Y[branch_i,branch_j].real * np.sin(theta)) - Y[branch_i,branch_j].imag * np.cos(theta)) * \
                                                     c[2+2+5+k-1]
        d2_g4_d2_x_title[2*branch_i+1, 2*branch_i] = d2_g4_d2_x_title[2*branch_i, 2*branch_i+1]
        # d2_g4_d2_x_title[2*branch_i+1, 2*branch_i+1] = 0
        d2_g4_d2_x_title[2*branch_i+1, 2*branch_i+1] = d2_g4_d2_x_title[2*branch_i+1, 2*branch_i+1] - \
                                                       2 * Y[branch_i,branch_j].real * c[2+2+5+k-1]
        d2_g4_d2_x_title[2*branch_i+1, 2*branch_j] = d2_g4_d2_x_title[2*branch_j, 2*branch_i+1]
        d2_g4_d2_x_title[2*branch_i+1, 2*branch_j+1] = (Y[branch_i,branch_j].real * np.cos(theta) + \
                                                        Y[branch_i,branch_j].imag * np.sin(theta)) * c[2+2+5+k-1]
        d2_g4_d2_x_title[2*branch_j+1, 2*branch_i] = d2_g4_d2_x_title[2*branch_i, 2*branch_j+1]
        d2_g4_d2_x_title[2*branch_j+1, 2*branch_i+1] = d2_g4_d2_x_title[2*branch_i+1, 2*branch_j+1]
        d2_g4_d2_x_title[2*branch_j+1, 2*branch_j] = d2_g4_d2_x_title[2*branch_j, 2*branch_j+1];
        d2_g4_d2_x_title[2*branch_j+1, 2*branch_j+1] = 0

    d2_g_dx_c[2*num_gen:len_x,2*num_gen:len_x] = d2_g4_d2_x_title
    # print("第" + str(num_iteration) + "次迭代的等式约束的Hessian矩阵与Lagrange乘子y乘积为：")
    # mat_out(d2_g_dx_c)

    # 合成Hessian矩阵
    H_ = -d2f_dx + d2h_dx_y + d2_g_dx_c - np.matmul(np.matmul(dg_dx, (L_Z - U_W)), dg_dx.T)
    # print("第" + str(num_iteration) + "次迭代的Hessian矩阵为：")
    # mat_out(H_)
    # print(np.linalg.det(H_))

    # df = pd.DataFrame(dg_dx)
    # writer = pd.ExcelWriter('A.xlsx')
    # df.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # writer.save()
    # writer.close()

    temp = np.matmul(np.matmul(dg_dx, (L_Z - U_W)), dg_dx.T)

    # 二、生成系数矩阵
    A = np.zeros([4*num_inequa+num_equa+len_x, 4*num_inequa+num_equa+len_x])
    A[0:4*num_inequa, 0:4*num_inequa] = np.eye(4*num_inequa, 4*num_inequa)
    A[0:num_inequa, num_inequa:2*num_inequa] = L_Z
    A[num_inequa:2*num_inequa, 4*num_inequa:4*num_inequa+len_x] = -dg_dx.T
    A[2*num_inequa:3*num_inequa, 3*num_inequa:4*num_inequa] = U_W
    A[3*num_inequa:4*num_inequa, 4*num_inequa:4*num_inequa+len_x] = dg_dx.T
    A[4*num_inequa:4*num_inequa+len_x, 4*num_inequa:4*num_inequa+num_equa+len_x] = np.concatenate((H_, dh_dx),axis=1)
    A[4*num_inequa:4*num_inequa+len_x, 4*num_inequa:4*num_inequa+num_equa+len_x]
    A[4*num_inequa+len_x:4*num_inequa+num_equa+len_x, 4*num_inequa:4*num_inequa+len_x] = dh_dx.T
    # print("第" + str(num_iteration) + "次迭代的系数矩阵为：")
    # mat_out(A)

    return A, dh_dx, dg_dx, H_, d2h_dx_y, d2_g_dx_c, d2f_dx, temp, L_Z, U_W