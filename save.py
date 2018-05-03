import cv2
from datetime import datetime as dt

def save(img,loc):
    t= dt.now()
    out_name = t.strftime('%I-%M-%S-%p-%d-%m')
    file_name = out_name + ".jpg"
    cv2.imwrite('report-images/' + loc + "/" + file_name , img )
