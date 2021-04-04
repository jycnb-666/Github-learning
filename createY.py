"""coding = utf-8"""
import numpy as np

def createY(N, B, num_node, num_branch):
    """生成并输出节点导纳矩阵"""
    Y = np.ones((num_node,num_node)) * (0+0j)      # 初始化节点导纳矩阵
    start = B[:,0].tolist()
    end = B[:,1].tolist()

    Z_s = B[:,6] + 1j * B[:,7],    # 串行支路阻抗
    Y_s = np.ones(num_branch) * (0+0j)

    for i in range(0,num_branch):
        if Z_s[0][i] != 0:
            Y_s[i] = 1/Z_s[0][i]
    Y_p = 1j * B[:,8]      # 并行支路导纳
    k_ratio = B[:,14]     # 变压器变比

    for i in range(0, num_branch):
        start[i] = int(start[i]-1)
        end[i] = int(end[i]-1)
        if B[i,5] == 0:             # 对应支路为输电线路
            Y[start[i],start[i]] = Y[start[i],start[i]] + Y_s[i] + Y_p[i]/2
            Y[end[i],end[i]] = Y[end[i],end[i]] + Y_s[i] + Y_p[i]/2
            Y[start[i],end[i]] = Y[start[i],end[i]] - Y_s[i]
            Y[end[i],start[i]] = Y[start[i],end[i]]
        elif B[i,5] == 1:           # 对应支路为变压器支路，这里采用pi形等值电路
            Y[start[i],start[i]] = Y[start[i],start[i]] + Y_s[i] / (k_ratio[i])**2
            Y[end[i],end[i]] = Y[end[i],end[i]] + Y_s[i]
            Y[start[i],end[i]] = Y[start[i],end[i]] - Y_s[i] / k_ratio[i]
            Y[end[i],start[i]] = Y[start[i],end[i]]
        else:
            print("支路类型输入错误，无法形成节点导纳矩阵！")
    for k in range(0,num_node):         # 节点对地导纳
        Y[k,k] = Y[k,k] + N[k,13] + 1j * N[k,14]

    print("该网络的导纳矩阵为：")
    for i in range(0,num_node):
        for j in range(0,num_node):
             print('({0.real:5.2f} + {0.imag:5.2f}i)'.format(Y[i][j]), end="\t\t")
        print()
    print()
    return Y
