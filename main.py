import cv2
from camera import camera
from camera import recorder
from camera import detector

if __name__ == '__main__':
    cam = camera.Camera(0)
    rec = recorder.Recorder('log.mp4')
    det = detector.Detector()
    while True:
        try:
            frame = cam.get()
            if frame is None:
                continue
            rec.put(frame)
            cv2.imshow('video', det.detect(frame))
            if cv2.waitKey(1) == 27:
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

    cam.quit()
    rec.quit()
    del cam
    del rec
