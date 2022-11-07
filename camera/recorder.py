import cv2
import threading
import queue
from camera import config


class Recorder:
    def __init__(self, fname):
        self.frames_queue = queue.Queue(5)
        self.fname = fname
        self.stop = threading.Event()
        self.recorder_proc = threading.Thread(target=self.recorder)
        self.recorder_proc.start()

    def recorder(self):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(self.fname, fourcc, config.VIDEO_FPS, config.VIDEO_RESOLUTION)
        while not self.stop.is_set():
            if not self.frames_queue.empty():
                frame = self.frames_queue.get()
                writer.write(frame)
        writer.release()

    def quit(self):
        self.stop.set()
        self.recorder_proc.join()

    def __del__(self):
        self.quit()
        del self.stop
        del self.frames_queue

    def put(self, frame):
        if not self.frames_queue.full():
            self.frames_queue.put(frame)
