import time
import torch
import cv2
import numpy as np
from PIL import Image,ImageDraw
import argparse
import os
from YoloObbTrack.Tracker import *

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, non_max_suppression_obb,  scale_polys)
from utils.rboxs_utils import rbox2poly

def save_image(image, address, num):
    pic_address = address + str(num) + '.jpg'
    cv2.imwrite(pic_address, image)

def track(opt):
    video_path      = opt.video_path
    video_save_path = opt.output
    video_fps       = opt.video_fps
    weights         = opt.weights   
    img_save_path   = opt.img_save_path
    track_type      = opt.track_type
    is_track_img    = opt.is_track_img
    track_img_path  = opt.track_img_path
    is_track_det_img= opt.is_track_det_img
    track_det_img_path = opt.track_det_img_path

    #Determine whether the image storage path exists, and create it if it does not exist
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    #Loading the detection model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half='False'
    model = DetectMultiBackend(weights, device=device, dnn=False)
    model.model.half() if half else model.model.float()


    capture = cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_video     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("The camera (video) cannot be read correctly. Please pay attention to whether the camera is installed correctly (whether the video path is filled in correctly).。")

    j=0
    while(True): 
        # Read a frame
        ref, frame = capture.read()
        if not ref:
            break
        j=j+1

        save_image(frame, img_save_path, j)
        print('Image save address：', img_save_path+ str(j) + '.jpg')
        
        images_name='{}.jpg'.format(j)
        PIL_image=Image.open(img_save_path+images_name)
        #Loading an image and detecting it
        dataset = LoadImages(img_save_path+images_name, img_size=640)
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model(im)
            # NMS
            pred = non_max_suppression_obb(pred, 0.3, 0.1, multi_label=True, max_det=1000)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                #Zoom to original size
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0s.shape)


        # Perform tracking detection and output tracking frames
        t1 = time.time()
        frame , rboxes2draws= update_tracker(frame,pred,PIL_image,im.shape[2:],im0s.shape,track_type)
        fps  = 1./(time.time()-t1)
        print("fps_track= %.2f"%(fps))
        frame = np.ascontiguousarray(frame)
        #Box to store traces
        if is_track_img:
            if not os.path.exists(track_img_path):
                os.makedirs(track_img_path)   
            save_image(frame, track_img_path, j)        

        #Draw a detection frame (black) on the tracking frame
        if is_track_det_img:
            if not os.path.exists(track_det_img_path):
                os.makedirs(track_det_img_path) 
            for poly in pred_poly:
                color = (0,0,0)
                draw = ImageDraw.Draw(PIL_image)
                draw.polygon(xy=list(poly), outline=color, width=1)
                del draw 
            PIL_image=np.ascontiguousarray(PIL_image)
            save_image(PIL_image, track_det_img_path, j)


        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out_video.write(frame)

        if c==27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out_video.release()
    cv2.destroyAllWindows()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='test_video/test/testup1.mp4')
    parser.add_argument('--output', type=str, default='test_video/output/output2.mp4')
    parser.add_argument('--img_save_path', type=str, default='test_video/test_data/ori_img/')
    parser.add_argument('--video_fps', type=int, default=10)
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/last.pt')
    parser.add_argument('--track_type', type=str, default='Byte_tracker')
    parser.add_argument('--is_track_img',action='store_true')
    parser.add_argument('--track_img_path', type=str, default='test_video/test_data/output_images/')
    parser.add_argument('--is_track_det_img', action='store_true')
    parser.add_argument('--track_det_img_path', type=str, default='test_video/test_data/output_images_det/')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt



if __name__ == "__main__":
    opt = parse_opt()
    track(opt)