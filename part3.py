import numpy as np
import cv2
import scipy
from matplotlib import pyplot


def myHarrisCornerDetector(video_path, K, Threshold):
    orig_video = cv2.VideoCapture(video_path)
    fourcc = orig_video.get(6)
    fps = orig_video.get(5)
    frameSize = (int(orig_video.get(3)), int(orig_video.get(4)))
    binary_video = cv2.VideoWriter('Vid1_Binary.avi', int(fourcc), fps, frameSize, isColor=True)
    while orig_video.isOpened():
        hasFrames, frame = orig_video.read()  # Capture frame-by-frame
        if hasFrames:  # hasFrames returns a bool,  If frame is read correctly - it will be True.
            frame = cv2.resize(frame, frameSize)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret2, thresholdFrames = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_video.write(thresholdFrames)  # write the binary frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        orig_video.release()
        binary_video.release()


def createCornerPlots(videoPath):
    orig_video = cv2.VideoCapture(videoPath)
    fourcc = orig_video.get(6)
    fps = orig_video.get(5)
    frameSize = (int(orig_video.get(3)), int(orig_video.get(4)))
    gray_video = cv2.VideoWriter('Vid2_Grey.avi', int(fourcc), fps, frameSize, isColor=False)
    while orig_video.isOpened():
        hasFrames, frame = orig_video.read()
        if hasFrames:  # hasFrames returns a bool, if frame is read correctly - it will be True
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_video.write(gray_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    orig_video.release()
    gray_video.release()


