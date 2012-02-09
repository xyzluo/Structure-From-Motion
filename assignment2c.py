import os
import pydot
import numpy as np
from PIL import Image
import tools

#must run assignment2b.py first to generate 'tree' file
matchscores = np.loadtxt('tree.out')
print (matchscores)
image_path = "D:\\code\\mycode\\mypy\\data\\3\\scene"
# path to save thumbnails (pydot needs the full system path)
path = os.getcwd() + "\\data\\temp\\" 

# list of filenames
imlist = tools.get_imlist(image_path)
nbr_images = len(imlist)

threshold = 0 # min number of matches needed to create link

g = pydot.Dot(graph_type='graph') # don't want the default directed graph 
for i in range(nbr_images):
    for j in range(i+1,nbr_images):
        if matchscores[i,j] > threshold:
            # first image in pair
            im = Image.open(imlist[i])
            im.thumbnail((100,100))
            filename = path+str(i)+'.png'
            im.save(filename) # need temporary files of the right size 
            g.add_node(pydot.Node(str(i),fontcolor='transparent',shape='rectangle',image=filename))

            # second image in pair
            im = Image.open(imlist[j])
            im.thumbnail((100,100))
            filename = path+str(j)+'.png'
            im.save(filename) # need temporary files of the right size 
            g.add_node(pydot.Node(str(j),fontcolor='transparent',shape='rectangle',image=filename)) 
            
            g.add_edge(pydot.Edge(str(i),str(j)))

g.write_png('largerimage.png')