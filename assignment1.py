import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tools

# Assumptioin: camera intrinsic parameters is [I]
# and there is no distortion
def compute_fundamental(xx1,xx2):
    """    
    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    this implementation is modified from Jan Erik Solem's book:
    <Programming Computer Vision with Python: Tools and algorithms for 
    analyzing images> and according to [Richard I. Hartley], to
    improve numeric stability, added input normalization output de-normalization.     
    """

    n = xx1.shape[1]
    if xx2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # normalize input 
    x1 = xx1[:2].copy()
    x2 = xx2[:2].copy()
    mean1 = x1.mean(axis=1,keepdims=True)
    mean2 = x2.mean(axis=1,keepdims=True)
    # mean1 *=0; mean2 *=0
    x1 -= mean1
    x2 -= mean2

    mean_dev1 = np.mean(np.sqrt(x1[0]**2 + x1[1]**2))
    mean_dev2 = np.mean(np.sqrt(x2[0]**2 + x2[1]**2))
    scale1 = np.sqrt(2.)/mean_dev1
    scale2 = np.sqrt(2.)/mean_dev2
    # scale1 =1;scale2=1

    x1*=scale1
    x2*=scale2
    x1 = np.vstack((x1,np.ones(x1.shape[1])))
    x2 = np.vstack((x2,np.ones(x2.shape[1])))

    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    
    #inverse of normalization
    T1 = np.eye(3)
    T2 = np.eye(3)

    T1[0,0] = scale1
    T1[1,1] = scale1
    T1[0,2] = -scale1*mean1[0]
    T1[1,2] = -scale1*mean1[1]
    
    T2[0,0] = scale2
    T2[1,1] = scale2
    T2[0,2] = -scale2*mean2[0]
    T2[1,2] = -scale2*mean2[1]

    F = np.dot(T2.T, np.dot(F.T,T1))
    return F/F[2,2]

if __name__ == "__main__":
    np.set_printoptions(precision=3)

    # read in csv and display
    from numpy import genfromtxt
    csv_data = genfromtxt('point_corresp.csv', delimiter=',')
    x1 = csv_data[1:,0:2].T
    x1 = np.vstack((x1,np.ones(x1.shape[1])))
    x2 = csv_data[1:,2:4].T
    x2 = np.vstack((x2,np.ones(x2.shape[1])))
    plt.figure(1)
    plt.subplot(121)
    plt.plot(x1[0,:],x1[1,:])
    plt.plot(x1[0,:],x1[1,:],'-ro')
    plt.subplot(122)
    plt.plot(x2[0,:],x2[1,:])
    plt.plot(x2[0,:],x2[1,:],'-bo')
    plt.show()

    # K = [[fx,0,cx],
    #      [0,fy,cy],
    #      [0, 0, 1]]
    # assume camera intrinsic is [I]
    K = np.eye(3) 
    # transfer pixel coord to normalized camera coord
    x1_mine = np.dot(np.linalg.inv(K),x1)
    x2_mine = np.dot(np.linalg.inv(K),x2)
    x1 = x1_mine[:2,:].T
    x2 = x2_mine[:2,:].T
    # findFundamentalMat does these:
    # 1. compute linear least square solution
    # 2. enforce result's rank to 2
    # since x1,x2 are now in camera coords, we can use 
    # findFundamentalMat to get essential matrix
    # the implementation shows same result as OpenCV implementation
    E1,_ = cv.findFundamentalMat(x1,x2)
    E = compute_fundamental(x1_mine,x2_mine)
    print (E-E1)

    # cv.recoverPose packed these steps:
    # 1. SVD decomposite E and return 4 possible poses, 
    #    with a unit t in 2 opposite directions, this is done
    #    in cv.decomposeEssentialMat()
    # 2. from possible poses find the one has positive depth.
    rr = cv.decomposeEssentialMat(E)
    _,R,t,_ = cv.recoverPose(E,x1,x2,K)    
    T = np.vstack((np.hstack((R,t)),[0,0,0,1]))
    print (T)
    np.savetxt('Homogeneous.out',T, fmt='%10.5f')

    # triangulation
    # camera pose
    Ml = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    Mr = np.hstack((R, t))
    # combine camera pose and intrinsic params
    Pl = np.dot(K,  Ml)
    Pr = np.dot(K,  Mr)
    point_4d_hom = cv.triangulatePoints(Pl, Pr, np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1))
    # normalize
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T
    print (point_3d)
    np.savetxt('triangulated.out',point_3d, fmt='%10.5f')
    
    # 
    # reproject back to 2d
    x2d_l = np.dot(Pl,point_4d)
    x2d_l_normalized = x2d_l/np.tile(x2d_l[-1,:],(3,1))
    x2d_r = np.dot(Pr,point_4d)
    x2d_r_normalized = x2d_r/np.tile(x2d_r[-1,:],(3,1))
    print (x2d_r_normalized[:2,:])
    # re-projected points should be close to original image points
    np.testing.assert_allclose(x2d_r_normalized[:2,:], x2.T, rtol=1e-5, atol=0)
    np.testing.assert_allclose(x2d_r_normalized[:2,:], x2.T, rtol=1e-5, atol=0)

    # 
    # plot constructed 3D points
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X,Y,Z = point_3d[:,0], point_3d[:,1],point_3d[:,2]
    # plot the points in 3D
    ax.plot(-X,Y,Z,label='3d points')
    plt.show()
    pass