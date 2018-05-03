
# <------ip address or link of live feed--->
# ip address or link of live stream to catch
ip_address = "http://192.168.43.1:8080/video"

# <--------distance_threshold is same as tolerance---->
# only difference is theshold is used in knn while tolerance is used in normal model

distance_threshold = 0.6

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
