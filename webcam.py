import time
import cv2.cv2 as cv2
from threading import Thread, Lock


class Webcam:

    def __init__(self, video_width=640, video_height=480):

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
        print("Webcam has been initialised.")

        self.color_frame, self.depth_frame = None, None

        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("Already started!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        time.sleep(0.7)
        while self.started:

            ret, c = self.cap.read()
            d = None

            self.read_lock.acquire()
            self.color_frame, self.depth_frame = c, d
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        color_frame, depth_frame = self.color_frame, self.depth_frame
        self.read_lock.release()
        return color_frame, depth_frame

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()