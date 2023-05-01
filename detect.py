import jetson.inference
import jetson.utils
import cv2

# initialize the camera
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

# load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# initialize the Centroid Tracker
class CentroidTracker():
    def __init__(self, maxDisappeared=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, centroids):
        # if there are no objects being tracked, register new objects
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i])
        # otherwise, match the centroids with the existing objects
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = jetson.utils.euclidean_distance(objectCentroids, centroids)

            # find the smallest value in each row and sort the row indexes based on the minimum values
            rows = D.min(axis=1).argsort()

            # find the smallest value in each column and sort the column indexes based on the minimum values
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = centroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if there are more centroids than existing objects, register new objects
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    self.register(centroids[row])
            # if there are more existing objects than centroids, mark the objects as disappeared
            else:
                for objectID in list(self.disappeared.keys()):
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # update the disappeared count for each object
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if an object has disappeared for too many frames, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        return self.objects

# initialize the Centroid Tracker
ct = CentroidTracker()

# process frames from the camera
while True:
    # capture a frame from the camera
    img, width, height = camera.CaptureRGBA(zeroCopy=1)

    # detect objects in the frame
    detections = net.Detect(img, width, height)

    # update the tracker with the new detections
    rects = []
    for detection in detections:
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        conf = detection.Confidence
        label = net.GetClassDesc(detection.ClassID)
        rects.append((left, top, right, bottom, conf, label))

    objects = ct.update(rects)

    # draw bounding boxes and labels around detected objects
    for (objectID, centroid) in objects.items():
        startX, startY, endX, endY, conf, label = rects[objectID]
        text = "{}: {:.2f}".format(label, conf)

        # draw the bounding box and label of the object
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(img, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # draw the centroid of the object
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # display the image
    jetson.utils.cudaDeviceSynchronize()
    jetson.utils.cudaDeviceSynchronize()
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection | {:d} FPS".format(int(1000.0 / net.GetNetworkTime())))

    # exit on ESC
    if display.IsClosed():
        break