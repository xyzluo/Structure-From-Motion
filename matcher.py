import cv2 as cv
import numpy as np
import _pickle as pickle

class Matcher(object):
    def __init__(self, data):
        self.names = []
        self.kps = []
        self.desc = []
        self.match_table = {}
        for k,v in data.items():
            self.names.append(k)          
            self.kps.append(v[0])
            self.desc.append(v[1])

    def match(self, scene_features, obj_id=0, obj_features=None):
        keypoints_obj = descriptors_obj = None
        if obj_features is None:
            keypoints_obj, descriptors_obj = self.kps[obj_id],self.desc[obj_id]
        else:
            keypoints_obj, descriptors_obj = obj_features

        keypoints_scene, descriptors_scene = scene_features

        # Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

        # apply the Lowe's ratio test
        ratio_thresh = 0.75 #0.5 #0.75
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        return (good_matches, keypoints_obj, keypoints_scene)

    def __builtMaxSpanningTree(self,mat,threshold=30):
        '''
        input
        mat: undirected graph presented in a adjacency matrix where images are vertices
        and number of matches are edges. 
        threshold: matches less than threshold is not stable,so ignored
        return value
        childLists: is a matrix presenting a spanning tree where row childLists[i] holds a list of 
        children of vertex[i]
        parentList: tree nodes are indexed as [0 ... n].  parentList is a list 
        where each element specifes index of a node's parent node.
        i.e: index_of_parent_of_node_i = parentList[i]
        '''
        n = mat.shape[0]
        parentList = -1*np.ones(n,dtype='int')
        parentList[0]=0 # default 1st image as tree root
        childLists = np.zeros((n,n))
        for i in range(n): #work through upper triangle
            for j in range(i+1,n):
                key = mat[i][j]
                if key < threshold:
                    continue
                if parentList[j]<0: #no parent, just add it
                    parentList[j] = i
                    childLists[i][j] = key
                else:
                    v=childLists[parentList[j]][j]#existing 
                    if key>v:
                        childLists[parentList[j]][j] = 0 #erase old
                        parentList[j] = i
                        childLists[i][j] = key # set new
        return childLists,parentList

    def __build_score_graph(self):
        nbr_images = len(self.kps)
        if (nbr_images<2): return None

        match_score = np.zeros((nbr_images,nbr_images))
        for i in range(nbr_images):
            for j in range(i+1,nbr_images):
                good_match,kp_obj,kp_scene = self.match((self.kps[i],self.desc[i]),obj_id=0,
                                        obj_features=(self.kps[j],self.desc[j]))
                match_score[i][j]= len(good_match)
                self.match_table[str(i)+str(j)] = (good_match,kp_obj,kp_scene)
                self.match_table[str(j)+str(i)] = (good_match,kp_scene,kp_obj)
        return match_score

    def getHomography(self,i,j):
        name = str(i)+str(j)
        if not self.match_table:
            raise ValueError('[{i},{j}] is not in match_table.')
        ma,keypoints_obj,keypoints_scene = self.match_table[name]
        obj = np.empty((len(ma),2), dtype=np.float32)
        scene = np.empty((len(ma),2), dtype=np.float32)
        for i in range(len(ma)):
            #Get keypoints from good matches
            obj[i,0] = keypoints_obj[ma[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[ma[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[ma[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[ma[i].trainIdx].pt[1]
        H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
        return (H,obj,scene)

    def getCameraMotion(self,i,j,K):
        name = str(i)+str(j)
        if not self.match_table or (not name in self.match_table):
            raise ValueError('[{i},{j}] is not in match_table.')
        ma,keypoints_obj,keypoints_scene = self.match_table[name]
        obj = np.empty((len(ma),2), dtype=np.float32)
        scene = np.empty((len(ma),2), dtype=np.float32)
        for i in range(len(ma)):
            #Get keypoints from good matches
            obj[i,0] = keypoints_obj[ma[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[ma[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[ma[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[ma[i].trainIdx].pt[1]
        E, _ =  cv.findEssentialMat(obj, scene, K, cv.RANSAC)
        _,R,t,_ = cv.recoverPose(E,obj,scene,K)
        return (R,t)       

    def spanning_tree(self, threshold=30):
        graph = self.__build_score_graph()
        return self.__builtMaxSpanningTree(graph,threshold=threshold)