"""coding = utf8"""
import numpy as np
from list_tran import list_tran
from mat_out import mat_out

def d_x(H_, dg_dx, dh_dx, L_Z, U_W, A, b, z, l, u, w, x_variable, y):
    """求解代数方程组，输出修正量"""
    (len_z, t_z) = z.T.shape
    (len_l, t_l) = l.shape
    (len_u, t_u) = u.shape
    (len_w, t_w) = w.T.shape
    (len_x, t_x) = np.array(x_variable).shape
    (len_y, t_y) = y.T.shape
    b_z = b[0:len_z, :]
    b_l = b[len_z:(len_z+len_l), :]
    b_w = b[(len_z+len_l):(len_z+len_l+len_w), :]
    b_u = b[(len_z+len_l+len_w):(len_z+len_l+len_w+len_u), :]
    b_x = b[(len_z+len_l+len_w+len_u):(len_z+len_l+len_w+len_u+len_x), :]
    b_y = b[(len_z+len_l+len_w+len_u+len_x):(len_z+len_l+len_w+len_u+len_x+len_y), :]

    # 单位修正量
    block = np.concatenate((np.concatenate((H_, dh_dx), axis=1),
                            np.concatenate((np.array(list_tran(dh_dx)), np.zeros([len_y,len_y])), axis=1)),
                            axis=0)
    del_xy = np.matmul(np.linalg.inv(block),np.concatenate((b_x,b_y),axis=0))
    # blo_b = (block[:][23]).reshape(1,len(block[:][0]))
    # mat_out(blo_b)
    # mat_out(del_xy)
    del_x = del_xy[0:len_x]
    del_y = del_xy[len_x:(len_x+len_y)]
    del_l = b_l + np.matmul(dg_dx.T,del_x)
    del_u = b_u - np.matmul(dg_dx.T,del_x)
    del_z = b_z - np.matmul(L_Z,del_l)
    del_w = b_w - np.matmul(U_W,del_u)

    # 修正步长
    alpha_p = 1
    alpha_d = 1
    for k in range(0,len(l)):
        if del_l[k,0]<0 and (-l[k,0]/del_l[k,0])<alpha_p:
            alpha_p = -l[k,0]/del_l[k,0]
        if del_u[k,0]<0 and (-u[k,0]/del_u[k,0])<alpha_p:
            alpha_p = -u[k,0]/del_u[k,0]
        if del_z[k,0]<0 and (-z.T[k,0]/del_z[k,0])<alpha_p:
            alpha_d = -z.T[k,0]/del_z[k,0]
        if del_w[k,0]<0 and (-w.T[k,0]/del_w[k,0])<alpha_p:
            alpha_d = -w.T[k,0]/del_w[k,0]

    # 求取步长
    alpha_p = 0.9995 * alpha_p
    alpha_d = 0.9995 * alpha_d

    # 求取修正量
    delta_x = np.concatenate((alpha_d*del_z, alpha_p*del_l, alpha_d*del_w,
                               alpha_p*del_u, alpha_p*del_x, alpha_d*del_y),axis=0)

    return delta_x