from typing import Dict
from .base import Basemodel
from ..libs.trt.inference import model_inference
import cv2
import numpy as np


class RTMPose(Basemodel):
    
    def __init__(self):
        self.conf_thres = 0.3
        self.iou_thres = 0.45

        self.input_layer = list(self.inputs.keys())
        if len(self.input_layer) > 1:
            print(f'This model gets {len(self.input_layer)} nodes')

        self.output_layer = list(self.outputs.keys())
        if len(self.output_layer) > 1:
            print(f'This model outputs {len(self.output_layer)} nodes')

    def preprocess(self, input_datas) -> Dict[int, np.ndarray]:
        self.img = input_datas['img']
        self.bboxes = input_datas['bboxes']
        for key in self.input_layer:
            input_attribute = self.inputs.get(key)
            self.input_size = [input_attribute.width, input_attribute.height]
            preprocessed_data = []
            centers = []
            scales = []
            for bbox in self.bboxes:
                # getBBoxCenterScale
                center, scale = self.bbox_xyxy2cs(bbox)
                # TopDownAffine
                img, scale = self.top_down_affine(self.input_size, scale, center, self.img)

                # toTensor
                mean = np.array([123.675, 116.28, 103.53])
                std = np.array([58.395, 57.12, 57.375])
                img = (img - mean) / std

                image = np.expand_dims(img.astype(np.float32), axis=0)
                # Convert the image to row-major order:
                data = np.ascontiguousarray(image)
                if input_attribute.format == 'nchw':
                    data = data.transpose(0,3,1,2)
                preprocessed_data.append(data)
                centers.append(center)
                scales.append(scale)

        return np.concatenate(preprocessed_data, axis=0), centers, scales

    
    def bbox_xyxy2cs(self, bbox, padding=1.25):
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]
        
        # get bbox center and scale
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale

    def fix_ratio(self, bbox_scale, aspect_ratio):
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale
    
    def get_warp_matrix(self, center, scale, rot=0., output_size=(256, 192) , shift=(0., 0.), inv=False):
        
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    def _rotate_point(self, pt, angle_rad):
        
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt


    def _get_3rd_point(self, a, b):
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def top_down_affine(self, input_size, bbox_scale, bbox_center, img):
        w, h = input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        bbox_scale = self.fix_ratio(bbox_scale, aspect_ratio=w / h)

        # get the affine matrix
        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self.get_warp_matrix(center, scale, rot, output_size=(w, h))

        # do affine transform
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        return img, bbox_scale

    def postprocess(self, inference_results, center, scale, simcc_split_ratio=2.0):
        simcc_x, simcc_y = inference_results.get('simcc_x'), inference_results.get('simcc_y')
        simcc_x, simcc_y = simcc_x.reshape((-1, 14, 384)), simcc_y.reshape((-1, 14, 512))
        out1, out2 = self.decode(simcc_x, simcc_y, simcc_split_ratio)
        keypoints, scores = out1[:len(self.bboxes)], out2[:len(self.bboxes)]
        # print(np.array(keypoints).shape, np.array(center).shape)
        # print(center)
        keypoints = keypoints[:len(scale)] / self.input_size * np.expand_dims(scale,1) + np.expand_dims(center,1) - np.expand_dims(scale,1) / 2
        return keypoints, scores

    def get_simcc_maximum(self, simcc_x, simcc_y):
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        # get maximum value locations
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        # get maximum value across x and y axis
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        # reshape
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

        return locs, vals

    def decode(self, simcc_x, simcc_y, simcc_split_ratio):
        
        keypoints, scores = self.get_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio
        return keypoints, scores

    def run(self, input_data, batch_size, **kwargs):
        """Todo"""
        pre_result, center, scale = self.preprocess(input_data)
        data_len = len(pre_result)
        out_keypoints = np.zeros((data_len, 14, 2), np.float32)
        out_scores = np.zeros((data_len, 14), np.float32)
        
        num_batches = int(data_len / batch_size)
        num_left = int(data_len % batch_size)
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data = {self.input_layer[0] : pre_result[s:e]}

            inference_result = model_inference(cls=self.__class__, preprocess_result=batch_data)
            keypoints, scores = self.postprocess(inference_result, center[s:e], scale[s:e])
            out_keypoints[s:e] = keypoints
            out_scores[s:e] = scores
        if e < data_len:
            batch_data = {self.input_layer[0] : pre_result[e:]}
            inference_result = model_inference(cls=self.__class__, preprocess_result=batch_data)
            keypoints, scores = self.postprocess(inference_result, center[e:], scale[e:])
            out_keypoints[e:] = keypoints[:num_left]
            out_scores[e:] = scores[:num_left]

        return out_keypoints, out_scores  