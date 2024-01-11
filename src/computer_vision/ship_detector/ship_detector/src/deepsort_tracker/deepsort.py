import os
import sys
import torch
import torchvision
import numpy as np
from scipy.stats import multivariate_normal

from src.deepsort_tracker.deep_sort.application_util import preprocessing as prep
from src.deepsort_tracker.deep_sort.deep_sort import nn_matching
from src.deepsort_tracker.deep_sort.deep_sort.detection import Detection
from src.deepsort_tracker.deep_sort.deep_sort.tracker import Tracker

from src.deepsort_tracker.siamese_net import SiameseNetwork

def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


# Deep sort needs the format `top_left_x, top_left_y, width,height
def format_yolo_output(out_boxes):
    for b in range(len(out_boxes)):
        out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2] / 2
        out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3] / 2
    return out_boxes


def pre_process(frame, detections):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()])

    crops = []
    for d in detections:

        for i in range(len(d)):
            if d[i] < 0:
                d[i] = 0

        img_h, img_w, img_ch = frame.shape

        xmin, ymin, w, h = d

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))

        try:
            crop = frame[ymin:ymax, xmin:xmax, :]
            crop = transforms(crop)
            crops.append(crop)
        except:
            continue

    crops = torch.stack(tuple(crops))

    return crops


class DeepsortWrapper:
    def __init__(self, model=None):
        # curr_dir = os.path.dirname(os.path.realpath(__file__))
        # print("Current dir: ", curr_dir)
        sys.path.insert(0, '.')

        # Loading this encoder is slow, should be done only once.
        self.encoder = SiameseNetwork()
        self.encoder.load_state_dict(torch.load(model))

        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()
        print("Deep sort model loaded from path: ", model)

        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", .5, 100)
        self.tracker = Tracker(self.metric)

        self.gaussian_mask = get_gaussian_mask().cuda()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()])

    def reset_tracker(self):
        self.tracker = Tracker(self.metric)

    def extract_features_only(self, frame, coords):

        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0

        img_h, img_w, img_ch = frame.shape

        xmin, ymin, w, h = coords

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))

        crop = frame[ymin:ymax, xmin:xmax, :]
        # crop = crop.astype(np.uint8)

        # print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)

        crop = self.transforms(crop)
        crop = crop.cuda()

        gaussian_mask = self.gaussian_mask

        input_ = crop * gaussian_mask
        input_ = torch.unsqueeze(input_, 0)

        features = self.encoder.forward_once(input_)
        features = features.detach().cpu().numpy()

        corrected_crop = [xmin, ymin, xmax, ymax]

        return features, corrected_crop

    def run_deep_sort(self, frame, out_scores, out_boxes, obj_classes):

        if len(out_boxes) == 0:
            self.tracker.predict()
            # print('No detections')
            # trackers = self.tracker.tracks
            # return trackers
            return self.tracker, []

        detections = np.array(out_boxes)
        
        # features = self.encoder(frame, detections.copy())

        processed_crops = pre_process(frame, detections).cuda()
        processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        dets = [Detection(bbox, score, feature, obj_classes)
                for bbox, score, feature, obj_classes in zip(detections, out_scores, features, obj_classes)]

        outboxes = np.array([d.tlwh for d in dets])

        outscores = np.array([d.confidence for d in dets])
        indices = prep.non_max_suppression(outboxes, 0.8, outscores)

        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets
