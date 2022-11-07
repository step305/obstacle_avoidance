import cv2
import threading
import queue
from camera import config


class Camera:
    def __init__(self, dev_id=0):
        self.frames_queue = queue.Queue(5)
        self.stop = threading.Event()
        self.dev_id = dev_id
        self.streamer_thread = threading.Thread(target=self.streamer,
                                                daemon=True)
        self.streamer_thread.start()

    def streamer(self):
        cap = cv2.VideoCapture(self.dev_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, config.VIDEO_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_RESOLUTION[1])
        while not self.stop.is_set():
            ret, frame = cap.read()
            if ret:
                if not self.frames_queue.full():
                    self.frames_queue.put(frame)
        cap.release()

    def quit(self):
        self.stop.set()
        self.streamer_thread.join()

    def __del__(self):
        self.quit()
        del self.frames_queue
        del self.stop

    def get(self):
        if not self.frames_queue.empty():
            return self.frames_queue.get()
        else:
            return None
