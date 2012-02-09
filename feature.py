import cv2 as cv
import numpy as np
import os
import os.path
import _pickle as pickle
import tools

# Feature extractor
def extract_features(image, size=32):
    try:
        detector = cv.xfeatures2d_SURF.create(hessianThreshold=400)#400,800
        kps, dsc = detector.detectAndCompute(image, None)
        # print (len(kps)) # == dsc.size/64
        # print (dsc.size)
        needed_size = (size * 64)
        if dsc.size < needed_size:
            # if less than 32 descriptors then padding zeros
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

    except cv.error as e:
        print ('Error: ', e)
        return None

    return (kps,dsc)

def extract_features_from_file(image_path,size=32):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE) 
    return extract_features(image,size)

def build_feature_db(images_path, pickled_db_file):
    imgfiles = tools.get_imlist(images_path)
    result = {}
    for f in imgfiles:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features_from_file(f)
    
    # save everything (key points, descripters) in pickled file
    with open(pickled_db_file, 'wb') as fp:
        pickle.dump(result, fp)