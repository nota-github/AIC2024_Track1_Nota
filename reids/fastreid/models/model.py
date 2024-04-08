
from typing import Dict, List
from .base import Basemodel
from ..libs.trt.inference import model_inference
import cv2
import numpy as np
class ReID(Basemodel):
    
    def __init__(self):

        self.input_layer = list(self.inputs.keys())
        if len(self.input_layer) > 1:
            print(f'This model gets {len(self.input_layer)} nodes')

        self.output_layer = list(self.outputs.keys())
        if len(self.output_layer) > 1:
            print(f'This model outputs {len(self.output_layer)} nodes')
        
        self.output_attribute = self.outputs.get(self.output_layer[0])
        self.feature_dim = self.output_attribute.shape[1]

    def preprocess(self, image, dets, is_pad=False) -> Dict[int, np.ndarray]:
        preprocessed_data = {}
        dets = np.array(dets)

        for key in self.input_layer:
            input_attribute = self.inputs.get(key)
            h, w = input_attribute.height, input_attribute.width

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if is_pad:
                images = np.asarray([self.extract_image_patch_pad(image, bbox, (h,w)) for bbox in dets[:, :4]])
            else:
                images = np.asarray([self.extract_image_patch(image, bbox, (h,w)) for bbox in dets[:, :4]])

            images = images.astype(np.float32)
            data = np.ascontiguousarray(images)
            try:
                if input_attribute.format == 'nchw':
                    data = data.transpose(0,3,1,2)
            except:
                print(data)
            preprocessed_data[key] = data
        return preprocessed_data
    
    def extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray / The full image.
        bbox : ndarray / The bounding box in format (x, y, x, y) : top-left, bottom-right
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        # if patch_shape is not None:  # 진행 여부에 따른 결과 차이 보기 / BoTSORT에선 진행 X
        #     # correct aspect ratio to patch shape
        #     bbox = self.xyxy2tlwh(bbox)
        #     target_aspect = float(patch_shape[1]) / patch_shape[0]
        #     new_width = target_aspect * bbox[3]
        #     bbox[0] -= (new_width - bbox[2]) / 2
        #     bbox[2] = new_width
        # else:
        #     # convert to top left, bottom right
        #     bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
    
    def extract_image_patch_pad(self, image, bbox, patch_shape):
        # if patch_shape is not None:  # 진행 여부에 따른 결과 차이 보기 / BoTSORT에선 진행 X
        #     # correct aspect ratio to patch shape
        #     bbox = self.xyxy2tlwh(bbox)
        #     target_aspect = float(patch_shape[1]) / patch_shape[0]
        #     new_width = target_aspect * bbox[3]
        #     bbox[0] -= (new_width - bbox[2]) / 2
        #     bbox[2] = new_width
        # else:
        #     # convert to top left, bottom right
        #     bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]

        padded_img = np.ones(patch_shape[0], patch_shape[1], 3) * 114
        r = min(patch_shape[0] / image.shape[0], patch_shape[1] / image.shape[1])
        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        )
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img

        return padded_img
    
    def xyxy2tlwh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [top-left, top-left, width, height] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height

        return y
    
    def postprocess(self, inference_results):
        for key in self.output_layer:
            inference_data = inference_results.get(key)
            features = inference_data.reshape((-1, self.feature_dim))
            features[np.isinf(features)] = 1.0
            features = self.normalize(features)

            return features
    
    def normalize(self, features):
            # Normalize feature to compute cosine distance
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / norm
            return features

    def run(self, input_data, detections, batch_size, **kwargs):
        """Todo"""
        pre_result = self.preprocess(input_data, detections, kwargs.get('is_pad'))[self.input_layer[0]]

        out = np.zeros((len(pre_result), self.feature_dim), np.float32)
        data_len = len(out)
        num_batches = int(data_len / batch_size)
        num_left = int(data_len % batch_size)

        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data = {self.input_layer[0] : pre_result[s:e]}
            inference_result = model_inference(cls=self.__class__, preprocess_result=batch_data)
            post_result = self.postprocess(inference_result)
            out[s:e] = post_result
        if e < len(out):
            batch_data = {self.input_layer[0] : pre_result[e:]}
            inference_result = model_inference(cls=self.__class__, preprocess_result=batch_data)
            post_result = self.postprocess(inference_result)
            out[e:] = post_result[:num_left]

        return out
    
    
    # def imshow_nan():
    #     """
    #     https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/fast_reid/fast_reid_interfece.py#L131 참고
    #     """
    #     nans = np.isnan(np.sum(feat, axis=1))
    #     if np.isnan(feat).any():
    #         for n in range(np.size(nans)):
    #             if nans[n]:
    #                 # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
    #                 patch_np = patches_[n, ...]
    #                 patch_np_ = torch.unsqueeze(patch_np, 0)
    #                 pred_ = self.model(patch_np_)

    #                 patch_np = torch.squeeze(patch_np).cpu()
    #                 patch_np = torch.permute(patch_np, (1, 2, 0)).int()
    #                 patch_np = patch_np.numpy()

    #                 plt.figure()
    #                 plt.imshow(patch_np)
    #                 plt.show()

