import cv2

def look_for_new_grid(frame):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame, 50, 150)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('canny', canny)


def main_grid_detector_video(display=False):
    iso_grid = False
    grid = None
    cap = cv2.VideoCapture(1)

    while "user does not exit":
        # Capture frame-by-frame
        _, frame = cap.read()

        look_for_new_grid(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ord('q')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

