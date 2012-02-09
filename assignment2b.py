import os
import os.path
import cv2 as cv
import numpy as np
import _pickle as pickle
import copyreg
import feature
import matcher as mtr
import tools
from timeit import default_timer as timer

scene_path = "./data/3/scene/"
scene_db = 'scene_features_db.pck'

def picture_tree():
    '''
    Each scene picture is overlapped with at lease one other picture.
    So there exists a single graph connecting all pictures, using number 
    of matched features as edge.
    This method does:
    1. build a graph and its spanning tree using any picture as root.
    2. get a chain of pictures from root picture to current selected picture
    3. calculate camera motion from picture to picture
    4. show location of current picture relative to root picture.
    5. identify object(s) in current picture. (see assinment2a.py)
    '''

    # load scene data
    if not os.path.isfile(scene_db):
        feature.build_feature_db(scene_path,scene_db)
    data = None
    with open(scene_db,'rb') as fp:
        data = pickle.load(fp)

    # get list of images from root to image
    ma = mtr.Matcher(data)
    tree, plist = ma.spanning_tree()
    print(tree)
    np.savetxt('tree.out',tree)
    # test an arbitary image[i],0<=i<=8
    i=8
    ll=[i]
    while plist[i]!=i:
        i=plist[i]
        ll.insert(0,i)
    print (ll)

    if (len(ll)<2): #do nothing if only root is in list
        return

    K = tools.camera_calibration(2160)
    T_final=np.diagflat(np.ones(4))    
    for i in range(len(ll)-1):
        R,t = ma.getCameraMotion(ll[i],ll[i+1],K)
        T = np.vstack((np.hstack((R,t)),[0,0,0,1.]))
        # print (T)
        T_final =  np.matmul(T,T_final)
        # H,kp1,kp2 = ma.getHomography(i,i+1)
        # print (H)
        # retval,R,T,N = cv.decomposeHomographyMat(H,K)
        # print (R,'\n',T,'\n',N)
        # kp1 and kp2 must be CV_32FC2
        # possible_solution = cv.filterHomographyDecompByVisibleRefpoints(R,N,kp1.reshape(-1,1,2),kp2.reshape(-1,1,2))
    print ('current camera pose:')
    print (T_final)

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    # make KeyPoint to be pickle-able firstly
    copyreg.pickle(cv.KeyPoint().__class__, tools.pickle_keypoints) 

    # rename all scene files so they are not ordered
    # tools.shuffle_scene_files(scene_path)

    picture_tree()
    cv.destroyAllWindows()

