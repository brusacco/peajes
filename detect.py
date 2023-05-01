import jetson.inference
import jetson.utils
import pyds

# load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# create the camera object
camera = jetson.utils.gstCamera(1280, 720, "0")

# create the display window
display = jetson.utils.glDisplay()

# create the DeepStream Tracker plugin
tracker = pyds.gst_tracker_multi_object_new()

# set the tracking algorithm to SORT
config_string = "tracker-width=1280 tracker-height=720 gpu-id=0 \
    ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_nvdcf.so \
    ll-config-file=/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_nvdcf.txt"
pyds.gst_tracker_multi_object_set_config(tracker, config_string)

while display.IsOpen():
    # capture the image from the camera
    img, width, height = camera.CaptureRGBA()

    # detect objects in the image
    detections = net.Detect(img, width, height)

    # create a list of object metadata for the DeepStream Tracker plugin
    object_meta_list = []
    for detection in detections:
        object_meta = pyds.NvDsObjectMeta()
        object_meta.object_id = detection.ClassID
        object_meta.rect_params.left = detection.Left
        object_meta.rect_params.top = detection.Top
        object_meta.rect_params.width = detection.Width
        object_meta.rect_params.height = detection.Height
        object_meta.confidence = detection.Confidence
        object_meta_list.append(object_meta)

    # perform object tracking using the DeepStream Tracker plugin
    frame_number = 0 # TODO: set the correct frame number
    pyds.gst_tracker_multi_object_track(tracker, frame_number, object_meta_list)

    # render the image and detections on the display window
    display.BeginRender()

    # draw the image on the display
    display.RenderOnce(img, width, height)

    # draw the object detection bounding boxes and labels
    for detection in detections:
        # get the object ID
        object_id = detection.ClassID

        # get the bounding box coordinates
        left = int(detection.Left)
        top = int(detection.Top)
        right = int(detection.Right)
        bottom = int(detection.Bottom)

        # draw the detection box
        display.DrawRect(left, top, right - left, bottom - top, thickness=2, color=(0, 255, 0))

        # draw the object ID label
        display.DrawText("{:d}".format(object_id), left + 5, top + 15, color=(0, 0, 255), font_size=12)

    # update the window title with the FPS
    display.EndRender()
    display.SetTitle("Object Detection | {:d} FPS".format(int(display.GetFPS())))
