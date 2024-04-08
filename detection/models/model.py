from typing import Dict
from .base import Basemodel
import cv2
import numpy as np
class NPNet(Basemodel):
    
    def __init__(self):
        self.classes = [
            'person'
        ]
        self.conf_thres = 0.25
        self.iou_thres = 0.45

        self.input_layer = list(self.inputs.keys())
        if len(self.input_layer) > 1:
            print(f'This model gets {len(self.input_layer)} nodes')

        self.output_layer = list(self.outputs.keys())
        if len(self.output_layer) > 1:
            print(f'This model outputs {len(self.output_layer)} nodes')

    def preprocess(self, image) -> Dict[int, np.ndarray]:
        preprocessed_data = {}

        for key in self.input_layer:
            input_attribute = self.inputs.get(key)
            self.input_size = [input_attribute.height, input_attribute.width]
            origin_h, origin_w, origin_c = image.shape
            self.origin_h, self.origin_w = origin_h, origin_w
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Calculate width and height and paddings
            r_w = self.input_size[1] / origin_w
            r_h = self.input_size[0] / origin_h
            if r_h > r_w:
                tw = self.input_size[1]
                th = int(r_w *  origin_h)
                tx1 = tx2 = 0
                ty1 = int((self.input_size[0] - th) / 2)
                ty2 = self.input_size[0] - th - ty1
            else:
                tw = int(r_h * origin_w)
                th = self.input_size[0]
                tx1 = int((self.input_size[1] - tw) / 2)
                tx2 = self.input_size[1] - tw - tx1
                ty1 = ty2 = 0
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0 
            # HWC to NHWC format
            image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order:
            data = np.ascontiguousarray(image)
            if input_attribute.format == 'nchw':
                data = data.transpose(0,3,1,2)
            preprocessed_data[key] = data
        return preprocessed_data
    
    def postprocess(self, inference_results):
        for key in self.output_layer:
            inference_data = inference_results.get(key)
            result = inference_data.reshape((len(self.classes)+4, -1))
            result = result.transpose()
            result = self.nms(result, self.conf_thres, self.iou_thres)
            result = self.normalize(result)
            result = self.scale_coords(result)
            # self.print_result(result)
            return result

    def nms(self, prediction, conf_thres, iou_thres):
        prediction = prediction[prediction[..., 4:].max(1) > conf_thres]
        boxes = self.xywh2xyxy(prediction[:, :4])
        res = self.non_max_suppression(boxes, prediction[:, 4:].max(1), iou_thres)
        result_boxes = []
        for r in res:
            tmp = np.zeros(6)
            j = prediction[r, 4:].argmax()
            tmp[0] = boxes[r][0].item()
            tmp[1] = boxes[r][1].item()
            tmp[2] = boxes[r][2].item()
            tmp[3] = boxes[r][3].item()
            tmp[4] = prediction[r][4:][j].item()
            tmp[5] = j
            result_boxes.append(tmp)
        return result_boxes
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y
    
    def non_max_suppression(self, boxes, scores, iou_thres):
        assert boxes.shape[0] == scores.shape[0]
        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
            filtered_indexes = np.where(ious > iou_thres)[0]
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)
    
    def compute_iou(self, box, boxes, box_area, boxes_area):
        assert boxes.shape[0] == boxes_area.shape[0]
        ys1 = np.maximum(box[0], boxes[:, 0])
        xs1 = np.maximum(box[1], boxes[:, 1])
        ys2 = np.minimum(box[2], boxes[:, 2])
        xs2 = np.minimum(box[3], boxes[:, 3])
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        unions = box_area + boxes_area - intersections
        ious = intersections / unions
        return ious
    
    def scale_coords(self, coords):
        if len(coords)==0:
            return coords
        gain = min(self.input_size[0] / self.origin_h, self.input_size[1] / self.origin_w)  # gain  = old / new
        pad = (self.input_size[0] - self.origin_w * gain) / 2, (self.input_size[1] - self.origin_h * gain) / 2  # wh padding
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords)
        return coords

    def clip_coords(self, coords):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, self.origin_w)  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, self.origin_h)  # y1, y2
    
    def normalize(self, boxes):
        if not boxes:
            return boxes
        np_boxes = np.array(boxes)

        if np.all(np_boxes[:,:4] <= 1.0):
            # restore result
            for box in boxes:
                box[0] *= self.input_size[1]
                box[1] *= self.input_size[0]
                box[2] *= self.input_size[1]
                box[3] *= self.input_size[0]
            return np.array(boxes)
        
        return np.array(boxes)
    
    def print_result(self, result_label):
        print("--------------------------------------------------------------")
        if result_label == []:
                print(' - Nothing Detected!')
        else:
            for i, label in enumerate(result_label):
                detected = str(self.classes[int(label[5])])
                conf_score = label[4]
                x1, y1, x2, y2 = label[0], label[1], label[2], label[3]
                print(' - Object {}'.format(i+1))
                print('     - CLASS : {}'.format(detected))
                print('     - SCORE : {:5.4f}'.format(conf_score))
                print('     - BOXES : {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(x1,y1,x2,y2))
        print("--------------------------------------------------------------\n")