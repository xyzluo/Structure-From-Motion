import os
import cv2 as cv
import numpy as np
import scipy
import shutil

EXT = ['jpg', 'jpeg', 'JPG', 'JPEG', 'gif', 'GIF', 'png', 'PNG']

def get_imlist(path):
    """    Returns a list of filenames for 
        all [jpg,bmp,tiff,png] images in a directory. """
    # return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    return [os.path.join(path, f) for f in sorted(os.listdir(path))  if any(f.endswith(ext) for ext in EXT)]

def pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv.resize(image, (hashSize + 1, hashSize))

	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]

	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h

def getHistogram(image):
    '''
    extract a 3D RGB color histogram from the image,
    using 8 bins per channel, normalize, and update
    the index
    '''
    hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    cv.normalize(hist,hist)
    hist = hist.flatten()
    return hist

def rotationMatrixToEulerAngles(R) :
    '''
    Calculates rotation matrix to euler angles
    '''
    sy = np.math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.math.atan2(R[2,1] , R[2,2])
        y = np.math.atan2(-R[2,0], sy)
        z = np.math.atan2(R[1,0], R[0,0])
    else :
        x = np.math.atan2(-R[1,2], R[1,1])
        y = np.math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def getEulerFromHomograyphy(H):
    '''
    get Euler angles x-y-z in degree from homography matrix
    '''
    norm = np.math.sqrt(H[0,0]**2 + H[1,0]**2+H[2,0]**2)
    H/=norm
    c1= H.transpose()[0]
    c2= H.transpose()[1]
    c3= np.cross(c1,c2)
    R =  np.zeros_like(H)
    R[0]=c1;R[1]=c2;R[2]=c3
    R = R.transpose()
    # SVD decomp
    u,_,vh = np.linalg.svd(R)
    R = np.matmul(u,vh)
    x,y,z = rotationMatrixToEulerAngles(R)*180.0/3.14159    
    return (x,y,z)

def cos_cdist(self, vector):
    # getting cosine distance between search image and images database
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

def polygon_angles(points):
    '''
    points are polygon vertex in counter-clock or clock wise sequence.
    '''
    angles = np.zeros(len(points))
    for i in range(len(points)):
        p1 = points[i]
        ref = points[i - 1]
        p2 = points[i - 2]
        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]
        # Use dotproduct to find angle between vectors
        # This always returns an angle between 0, pi
        numer = (x1 * x2 + y1 * y2)
        denom = np.math.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        angle = np.math.acos(numer / denom)
        # cross sign 
        if not cross_sign(x1, y1, x2, y2): # outer angle
            angle = np.pi*2 - angle
        angles[i] = angle
    return angles

def cross_sign(x1, y1, x2, y2):
    '''   
    sign of cross product of 2 vectors (x1,y1) and (x2,y2)
    return True if cross is positive, else False
    '''
    return x1 * y2 > x2 * y1

def show_img(img,title='image'):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    height = int(1024*img.shape[1]/img.shape[0])
    cv.resizeWindow(title,1024,height)    
    cv.imshow(title,img)
    cv.waitKey(500)
    # plt.imshow(img)
    # plt.show()

def getImageCorner(img_object):
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    return obj_corners

def shuffle_scene_files(path):
    '''
    shuffle image file names to simulate unordered images
    keep the 1st (200.jpg) unchanged for easy debugging
    '''
    orig_dir = os.getcwd();os.chdir(path)
    lf = get_imlist('.')
    # save 1st file, maybe 200.jpg
    tempName = '.\\temp\\'+ os.path.basename(lf[0])
    shutil.move(lf[0],tempName)
    lf = lf[1:]
    new_names = np.random.permutation(lf)
    [shutil.move(f, f+'.tmp') for f in lf]
    lf = [os.path.join('.', f) for f in sorted(os.listdir('.'))  if any(f.endswith(ext) for ext in ['tmp'])]
    [shutil.move(f1,f2) for f1,f2 in zip(lf,new_names)]
    shutil.move(tempName,'.\\'+ os.path.basename(tempName))
    os.chdir(orig_dir) 
# def camera_calibration(size):
#     """
#     given a 16:9 image of resolution (row,col), return calibraion matrix of 
#     samsung s7 camera.
#     """
#     row,col = size
#     fx = 3221*col/4032
#     fy = 3228*row/2268
#     K = np.diag([fx,fy,1])
#     K[0,2] = 0.5*col
#     K[1,2] = 0.5*row
#     return K

def camera_calibration(size):
    """
    given 1:1 image resolution (size,size), return calibraion matrix of 
    samsung s7 camera.
    
    """
    fx = 2320*size/2160
    fy = 2320*size/2160
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*size
    K[1,2] = 0.5*size
    return K