"""格式化输出矩阵"""
import numpy as np
def mat_out(A):
    """保留4位有效数字空间，小数点后2位"""
    (x, m) = np.array(A).shape
    for i in range(0, x):
        for j in range(0, m):
            print('{0.real:4.2f}'.format(A[i][j]), end='\t')
        print()
    print()
    return None
