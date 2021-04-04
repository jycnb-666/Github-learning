import numpy as np
from mat_out import mat_out

# optimal power flow
A = np.array([[0,0,3],[4,0,0],[0,8,1]])
mat_out(A)

def change_col(A,p,q):
    """对调第p列和第q列"""
    (m,n) = A.shape     # 获取矩阵的行列
    temp = []
    for i in range(0,m):    # 列变换
        temp = A[i][p-1]
        A[i][p-1] = A[i][q-1]
        A[i][q-1] = temp
    print("矩阵列变换：第" + str(p) + "列 <——> 第" + str(q) + "列")
    return A

def change_row(A,p,q):
    """对调第p行和第q行"""
    (m,n) = A.shape     # 获取矩阵的行列
    temp = []
    for i in range(0,n):    # 行变换
        temp = A[p-1][i]
        A[p-1][i] = A[q-1][i]
        A[q-1][i] = temp
    print("矩阵行变换：第" + str(p) + "行 <——> 第" + str(q) + "行")
    return A

def find_max(A):
    """寻找矩阵每一行最大元素作为主元，返回主元所在列"""
    (m,n) = A.shape     # 获取矩阵的行列
    max_A = []      # 储存对应行最大元素的列索引
    a = 0
    for i in range(0,m):
        t = A[i][:].tolist()
        # print(t.index(max(t)))
        max_A.append(t.index(max(t)))
    return max_A

def diag_domin_inv(A):
    """将矩阵转换成主对角元占优矩阵并求逆，然后还原为原矩阵的逆阵"""
    (m,n) = A.shape     # 获取矩阵的行列
    max_A = find_max(A)
    for i in range(0,len(max_A)):
        if i == max_A[i]:
            continue
        change_col(A,i,max_A[i])
    A = np.linalg.inv(A)
    for i in range(0,len(max_A)):
        j = len(max_A)-i-1
        if j == max_A[j]:
            continue
        change_row(A,j,max_A[j])
    return A

B = diag_domin_inv(A)
mat_out(B)
mat_out(np.linalg.inv(A))