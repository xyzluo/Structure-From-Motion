import cv2 as cv
import numpy as np
import os
import os.path
import _pickle as pickle
import copyreg
import feature
import matcher as mtr
import tools
from timeit import default_timer as timer

# import argparse
# from imageio import imread
# import matplotlib.pyplot as plt
# import imagehash
# from PIL import Image

def identify_objects_in_scene():
    '''
    loop through object database and check which can be found 
    in scene image. result is shown in output window.
    Green dot means found and red dot means not found
    '''
    # print (os.getcwd())
    object_path = "./data/1/object"
    obj_db = 'object_features_db.pck'
    # object feature pre-process: 
    # build known objects' feature database.It's a one time job.
    if not os.path.isfile(obj_db):
        feature.build_feature_db(object_path, obj_db)
    data = None
    with open(obj_db,'rb') as fp:
        data = pickle.load(fp)
    ma = mtr.Matcher(data)

    # scene features process
    scene_path = "./data/1/scene/11.jpg"
    img_scene = cv.imread(scene_path, cv.IMREAD_GRAYSCALE)   
    tools.show_img(img_scene)
    scene_features = feature.extract_features(img_scene)
    
    files = tools.get_imlist(object_path)
    for idx in range(len(files)):
        img_object = cv.imread(files[idx],cv.IMREAD_GRAYSCALE)
        # find object in scene
        # start = timer()
        good_matches, keypoints_obj, keypoints_scene = ma.match(scene_features, obj_id=idx) 
        #end = timer(); print (str(len(good_matches)) + "match takes time: ", end-start)

        # Draw matches
        img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # Localize the object
        obj = np.empty((len(good_matches),2), dtype=np.float32)
        scene = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            # Get the keypoints from the good matches
            obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        Homo, _ =  cv.findHomography(obj, scene, cv.RANSAC)
        # print (Homo)

        # Get the object border
        obj_corners = tools.getImageCorner(img_object)
        # print ("obj_corners\n", obj_corners)
        scene_corners = cv.perspectiveTransform(obj_corners, Homo)
        target_position = np.array ([scene_corners[0,0,0],scene_corners[0,0,1]])
        # print ("scene_corners\n", scene_corners)
        # Draw border of object mapped in scene
        cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
            (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (255,0,0), 16)
        cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
            (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (255,0,0), 16)
        cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
            (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (255,0,0), 16)
        cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
            (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (255,0,0), 16)
        H_rev = cv.getPerspectiveTransform(scene_corners,obj_corners)

        # #=====================================================================================
        # # debug & other try

        # print ("scene_to_obj_homo\n", H_rev)
        # Transform1 = cv.warpPerspective(img_object,Homo,(img_object.shape[0],img_object.shape[1]))
        # # show_img(Transform1)
       
        # # cv.line(img_scene, (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])),\
        # #     (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])), (255,0,0), 16)
        # # cv.line(img_scene, (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])),\
        # #     (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])), (255,0,0), 16)
        # # cv.line(img_scene, (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])),\
        # #     (int(scene_corners[3,0,0]), int(scene_corners[3,0,1])), (255,0,0), 16)
        # # cv.line(img_scene, (int(scene_corners[3,0,0]), int(scene_corners[3,0,1])),\
        # #     (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])), (255,0,0), 16)
        # # show_img(img_scene)
        # Transform2 = cv.warpPerspective(img_scene,H_rev,(1*img_scene.shape[0],1*img_scene.shape[1]),cv.WARP_INVERSE_MAP)
        # restoredObject_img = Transform2[:img_object.shape[0],:img_object.shape[1]]
        # cv.waitKey(0)
        # cv.destroyWindow('1st');cv.destroyWindow('2nd')
        # cv.imwrite('1st.jpg',img_object)
        # cv.imwrite('2nd.jpg',restoredObject_img)

        # # hash1 = dhash(img_object)
        # # hash2 = dhash(restoredObject_img)
        # # print ("this is hashing ================")
        # # print(hash1)
        # # print(hash2)
        # # print ('hamming distance of local dhash ', hash1-hash2)
        # # print (" ================")

        # # h1 = imagehash.dhash(Image.fromarray(img_object))
        # # h2 = imagehash.dhash(Image.fromarray(restoredObject_img))
        # # print ("h1-h2",h1-h2)

        # # check eulor angle, assume picture was taken this way: no rx ry rz > 0.25*pi
        # rx,ry,rz = tools.getEulerFromHomograyphy(Homo)
        # print ("(rx,ry,rz)", rx,ry,rz)

        # #debugging end
        # #====================================================================================
        
        # validate Homography by checking project poloyon inner angles and color histograph
        angles = tools.polygon_angles(scene_corners.reshape((4,2)))*180/np.pi

        # compare histogram
        img_object_color = cv.imread(files[idx])
        img_scene_restore_color = cv.imread(scene_path)
        img_scene_restore_color = cv.warpPerspective(img_scene_restore_color,H_rev,(img_scene.shape[0],img_scene.shape[1]),cv.WARP_INVERSE_MAP)
        img_scene_restore_color = img_scene_restore_color[:img_object.shape[0],:img_object.shape[1]]
        hist1 = tools.getHistogram(img_object_color)
        hist2 = tools.getHistogram(img_scene_restore_color)
        dist = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        # print ("histogram correlation:", dist)
        # scene_corners_restore = cv.perspectiveTransform(scene_corners, H_rev)
        # print ("scene_corners_restore\n", scene_corners_restore)        

        # draw found or not result on matches image
        font = cv.FONT_HERSHEY_DUPLEX
        if (dist<0.66 or angles.max()>175 or angles.min()<15 ):
            cv.putText(img_matches,"object not found",(200,1000),font,5,(100,100,255))
            cv.circle(img_matches,(100,1000), 63, (0,0,255), -1)
        else:
            cv.putText(img_matches,"object found", (200,1000),font,5, (255,100,100))
            cv.circle(img_matches,(100,1000), 63, (0,255,0), -1)
            print ("found object at position:\n", target_position)

        # Show & save matches image
        tools.show_img(img_matches) 
        cv.imwrite('result'+ str(idx) + '.png',img_matches)

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    # make KeyPoints to be pickle-able firstly
    copyreg.pickle(cv.KeyPoint().__class__, tools.pickle_keypoints) 
    identify_objects_in_scene()
    cv.destroyAllWindows()

