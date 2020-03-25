import cv2

'''
DANIEL KIGLI - TAU - VIDEO PROCESSING 2020
'''

# video manipulation using cv2
cap = cv2.VideoCapture('input.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
fps = 20.0
out_size = (320, 160)
out = cv2.VideoWriter('output.avi', fourcc, fps, out_size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:  # If we succeeded in reading a frame from the input video object
        frame = cv2.resize(frame, out_size)
        frame = cv2.flip(frame, 1)  # Flip horizontal

        # write the flipped frame
        out.write(frame)
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

