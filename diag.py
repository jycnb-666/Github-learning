"""coding = utf-8"""

import numpy as np

def get_diag(L):
    """提取主对角元，形成n*1的矩阵"""
    (L_m,L_n) = np.shape(L)
    if L_m!=L_n:
        print("该矩阵非对角阵，无法提取主元！")
        return None
    L_diag = []
    for i in range(0,L_m):
        for j in range(0,L_n):
            if i!=j:
                if L[i][j]!=0:
                    print("该矩阵非对角阵，无法提取主元！")
                    return None
                else:
                    continue
            else:
                L_diag.append(L[i,j])

    return np.array(L_diag).reshape(L_m,1)


def make_diag(L):
    """将1-D矩阵构造为对角矩阵"""
    (m,n) = np.shape(L)
    if (m==1 and n==1) or (m!=1 and n!=1):
        print("矩阵为(" + str(m) + "," + str(n) + ")维，无法构造对角阵！")
        return None
    elif m==1:
        L_diag = np.zeros([n,n])
        for i in range(0,n):
            L_diag[i][i] = L[0][i]
    else:
        L_diag = np.zeros([m,m])
        for i in range(0,m):
            L_diag[i][i] = L[i][0]
    return L_diag
