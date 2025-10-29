import math
from math import atan2, sin, cos
from . import const_value
import numpy as np
from scipy.spatial.transform import Rotation

def cart2geod(Xinput):
    X=Xinput[0]
    Y=Xinput[1]
    Z=Xinput[2]

    tolsq = 1e-10
    maxit = 10

    rtd   = 180/const_value.pi

    esq = (2-1/const_value.finv) / const_value.finv
    
    oneesq = 1-esq
    P=math.sqrt(X*X+Y*Y)

    if P > 1e-20:
        dlambda = math.atan2(Y,X) *rtd
    else:
        dlambda = 0

    if dlambda <0:
        dlambda = dlambda +360
    
    r=math.sqrt(P*P+Z*Z)

    if r>1e-20:
        sinphi=Z/r
    else :
        sinphi=0
    
    dphi = math.asin(sinphi)

    if r<1e-20:
        h=0
        return
    
    h = r-const_value.a*(1-sinphi*sinphi/const_value.finv)

    for i in range(maxit):
        sinphi = math.sin(dphi)
        cosphi = math.cos(dphi)

        N_phi = const_value.a/math.sqrt(1-esq*sinphi*sinphi)

        dP =P -(N_phi+h)*cosphi
        dZ=Z-(N_phi*oneesq+h)*sinphi

        h=h+(sinphi*dZ+cosphi*dP)
        dphi =dphi+(cosphi*dZ - sinphi*dP)/(N_phi+h)

        if dP*dP + dZ*dZ<tolsq:
            break

        if i==maxit-1:
            print('sth. wrong in cart2geod.')

    dphi=dphi*rtd
    geod=[]
    geod.append(dphi)
    geod.append(dlambda)
    geod.append(h)
    # print(geod)
    return geod

def cart2enu(X, dx):
    
    dtr=const_value.pi/180

    geod = cart2geod(X)
    # print(geod)
    cl = math.cos(geod[1]*dtr)
    sl = math.sin(geod[1]*dtr)
    cb = math.cos(geod[0]*dtr)
    sb = math.sin(geod[0]*dtr)

    east = -sl*   dx[0] +cl*   dx[1]+0
    north= -sb*cl*dx[0] -sb*sl*dx[1]+cb*dx[2]
    up   =  cb*cl*dx[0] +cb*sl*dx[1]+sb*dx[2]

    enu=[]
    enu.append(east)
    enu.append(north)
    enu.append(up)
    return enu

def enu2cart(X, enu):
    
    dtr=const_value.pi/180

    geod = cart2geod(X)
    # print(geod)
    cl = math.cos(geod[1]*dtr)
    sl = math.sin(geod[1]*dtr)
    cb = math.cos(geod[0]*dtr)
    sb = math.sin(geod[0]*dtr)

    #east = -sl*   dx[0] +cl*   dx[1]+0
    #north= -sb*cl*dx[0] -sb*sl*dx[1]+cb*dx[2]
    #up   =  cb*cl*dx[0] +cb*sl*dx[1]+sb*dx[2]

    dx0 = -sl*enu[0]-sb*cl*enu[1]+cb*cl*enu[2]
    dx1 =  cl*enu[0]-sb*sl*enu[1]+cb*sl*enu[2]
    dx2 =          0+   cb*enu[1]+   sb*enu[2]

    dx=[]
    dx.append(dx0)
    dx.append(dx1)
    dx.append(dx2)
    return dx

def hhmmss2sec(hhmmss):
    elem = hhmmss.split(':')
    sec = float(elem[0])*3600+float(elem[1])*60+float(elem[2])
    return sec

def Cen(X):
    dtr=const_value.pi/180

    geod = cart2geod(X)
    # print(geod)
    cl = math.cos(geod[1]*dtr)
    sl = math.sin(geod[1]*dtr)
    cb = math.cos(geod[0]*dtr)
    sb = math.sin(geod[0]*dtr)

    M = np.array([[-sl,cl,0],[-sb*cl,-sb*sl,cb],[cb*cl,cb*sl,sb]]).T
    return M

def rad2deg(l):
    ll = []
    for i in range(len(l)):
        ll.append(l[i]*180/math.pi)
    return ll

def deg2rad(l):
    ll = []
    for i in range(len(l)):
        ll.append(l[i]/180*math.pi)
    return ll

def m2att(R):
    att=[0,0,0]

    att[0] = math.asin(R[2, 1])
    att[1] = math.atan2(-R[2, 0], R[2, 2])
    att[2] = math.atan2(-R[0, 1], R[1, 1])

    return att

def att2m(att):
    sp = math.sin(att[0])
    cp=math.cos(att[0])
    sr = math.sin(att[1])
    cr =math.cos(att[1])
    sy=math.sin(att[2])
    cy= math.cos(att[2])
    R=np.array([[cy*cr - sy*sp*sr, -sy*cp, cy*sr + sy*sp*cr],\
        [sy*cr + cy*sp*sr, cy*cp, sy*sr - cy*sp*cr],\
            [-cp*sr, sp, cp*cr]])
    return R

def q2att(qnb):
    q0 = qnb[0]
    q1 = qnb[1]
    q2 = qnb[2]
    q3 = qnb[3]
    q11 = q0*q0
    q12 = q0*q1
    q13 = q0*q2
    q14 = q0*q3
    q22 = q1*q1
    q23 = q1*q2
    q24 = q1*q3
    q33 = q2*q2
    q34 = q2*q3
    q44 = q3*q3

    att=[0,0,0]
    att[0] = math.asin(2 * (q34 + q12))
    att[1] = math.atan2(-2 * (q24 - q13), q11 - q22 - q33 + q44)
    att[2] = math.atan2(-2 * (q23 - q14), q11 - q22 + q33 - q44)
    return att

def q2R(qnb):
    return att2m(q2att(qnb))

def alignRt(xyz0,xyz1):
    if len(xyz0)!=len(xyz1):
        raise Exception()
    N = len(xyz0)
    p1 = np.array([0.0,0.0,0.0])
    p2 = np.array([0.0,0.0,0.0])
    for i in range(N):
        p1 += np.array(xyz0[i])
        p2 += np.array(xyz1[i])
    p1 /= N
    p2 /= N

    W = np.zeros([3,3])
    for j in range(N):
        q1 = np.array(xyz0[j]) - p1
        q2 = np.array(xyz1[j]) - p2
        W += np.matmul(q1.reshape(3,1),q2.reshape(1,3))
    U, sigma, VT = np.linalg.svd(W)
    R= np.matmul(U,VT)
    t=p1-np.matmul(R,p2)
    return R,t

def R2ypr(R):
    n = R[0]
    o = R[1]
    a = R[2]

    y = atan2(n[1], n[0])
    p = atan2(-n[2], n[0] * cos(y) + n[1] * sin(y))
    r = atan2(a[0] * sin(y) - a[1] * cos(y), -o[0] * sin(y) + o[1] * cos(y))
    return np.array([y,p,r])

def ypr2R(ypr):
    y = ypr[0]
    p = ypr[1]
    r = ypr[2]

    Rz = np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
    Ry = np.array([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
    Rx = np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
        
    return np.matmul(np.matmul(Rz,Ry),Rx)

def FromTwoVectors(a,b):
    v0 = a/np.linalg.norm(a)
    v1 = b/np.linalg.norm(b)
    c = np.dot(v1,v0)
    axis = np.cross(v0,v1)
    s = math.sqrt((1+c)*2)
    invs = 1/s
    vec = axis*invs
    w = s* 0.5
    return Rotation.from_quat(np.array([vec[0],vec[1],vec[2],w])).as_matrix()


