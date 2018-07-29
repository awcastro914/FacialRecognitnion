Our facial recognition system consists of two main files. The first is the load_dadabase file which simply adds our training images 
into a readable format for our facial recognition system. The next file is face recognitionn.m which reads in our faces. To change
the face you want to read you must change a few lines. First you must change line 2 which is your test image. Next you must change line 99 
to be the correct directory. This will allow you to run the system on another image in our set. 

The pictures are located in pics\facePics. Here we have our test set which is orl_faces as well as our test directories which are test_set
and temp. Temp contains wall obstructed faces while temp contains cloth like obstruction.