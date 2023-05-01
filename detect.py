import jetson.inference
import jetson.utils

# initialize the camera
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

# load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# process frames from the camera
while True:
    # capture a frame from the camera
    img, width, height = camera.CaptureRGBA(zeroCopy=1)

    # detect objects in the frame
    detections = net.Detect(img, width, height)

    # print the number of detected objects
    print("Detected {:d} objects in image".format(len(detections)))

    # draw bounding boxes around detected objects
    for detection in detections:
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        print("Object: {:s} - Confidence: {:.2f}".format(detection.ClassID, detection.Confidence))
        img = jetson.utils.cudaToNumpy(img, width, height, 4)
        img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # display the image
    jetson.utils.cudaDeviceSynchronize()
    jetson.utils.cudaDeviceSynchronize()
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection | {:d} FPS".format(int(1000.0 / net.GetNetworkTime())))

    # exit on ESC
    if display.IsClosed() or (device != None and device.IsStreaming() == False):
        break
