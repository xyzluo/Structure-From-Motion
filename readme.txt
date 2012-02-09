The folder is structured like this:
./
|
|--assignment1.py
|
|--assignment2a.py  # 1st part of assignment2
|
|--assignment2b.py  # 2nd part of assignment2
|
|--assignment2c.py  # 3rd part of assignment2
|
|--matcher.py # used by assignment2
|
|--feature.py # used by assignment2
|
|--tools.py # used by all
|
|--readme.txt
|
|--review.txt
|
|--scene_features_db.pck  # program automatically generated cache file, 
|                           should delete and regenerate it when content of "./data/3/scene" is changed
|
|--object_features_db.pck # program automatically generated cache file, 
|                           should delete and regenerate it when content of "./data/1/object" is changed
|
|--_init_.py
|
|-- data
	|
	|-- 1  # image set for assignment2a.py
	|	|
	|	|--object
	|	|	|-- [1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg, 99.jpg] these are objects to be looking for in a scene image
	|	|
	|	|--scene
	|		|-- [11.jpg, 22.jpg ] these are scene pictures
	|	
	|	
	|---3  # image set for assignment2b.py
		|
		|--scene
			|-- [200.jpg, 201.jpg ... 208.jpg] Scene pictures for problem 2-b. The name orders doesn't matter because program checks all image pairs to find the highest match score.
				but user can still execute tools.shuffle_scene_files(scene_path) in assignment2b.py to shuffle the names. 
		

===================
To test assignment1
===================
>>> python assignment1.py
assignment1.py will draw the input data and result data, press 'q' to continue. It's the only interaction needed.
Result homogeneous transfermation and triangulate points will be printed to terminal and also saved to 
homogeneous.out and triangulated.out.

===================
To test assignment2
===================
>>> python assignment2a.py
The program will finish itself. As the goal is to identify what items from database are in the image. When executing, it will show if an object is found in a scene image, one by one. 
If it's found, a green dot is drawn on image otherwise a red dot is drawn. Also a polygon will be drawn arround the target. Its position is printed to terminal. The program will pause 500ms for each object.
The program will try to retrieve image features from object_features_db.pck, if it's not found, will create one. 
Intermediate results are saved as result*.png 

>>> python assignment2b.py
The program will finish itself. It creates a file 'tree.out' for next step building larger image.

>>> python assignment2c.py
There is no interaction for this program. This program generates a larger picture 'largerimage.png' using pydot. It also relies on result of assignment2b.py.
This part is divided from assignment2b because "dot.exe not found" may happen on another machine if pydot is not setup.
A pre-generated 'largerimage.png' is provided to show result on my machine. 

