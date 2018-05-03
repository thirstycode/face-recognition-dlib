
# <------ip address or link of live feed--->
# ip address or link of live stream to catch
ip_address = "http://192.168.43.1:8080/video"

# <----tolerance------>
# tolerance is strictness of recognizing Faces
# default value is 0.6
# lowering the tolerance can make face recognition accurate
# but lowering it can make something like no match if face in image are slightly different
# making it high can make recognize faces nicely but lowers the accuracy
# suggestion : only change tolerance value if you need to experiment
# keep it to default only
# If you are getting multiple matches for the same person, it might be that the people in your photos look very similar and a lower tolerance value is needed to make face comparisons more strict.
# The default tolerance value is 0.6 and lower numbers make face comparisons more strict:
# 0.6 is typical best performance.

tolerance = 0.6

# <----number_of_times_to_upsample –---------->
# number_of_times_to_upsample – How many times to upsample the image looking for faces. Higher numbers find smaller faces.
# default value is 1
# higher the number , more is the processing
# if you keep it as 2 or 3 , it will require 2x/3x time to process.
# but it'll show more faces, even smaller ones :D

number_of_times_to_upsample = 2

# <-------num_jitters------------->
# num_jitters – How many times to re-sample the face when calculating encoding. (Encoding is done on every face to get its geometry, every face has its own encoding ,so improving encoding can result into good results)
# Higher is more accurate, but slower (i.e. 100 is 100x slower)
# default value is 1
# i have made it 2 to make it better

num_jitters=2
