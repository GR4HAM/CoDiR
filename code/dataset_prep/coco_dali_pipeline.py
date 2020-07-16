from nvidia.dali.pipeline import Pipeline
import  nvidia.dali.ops as ops
import nvidia.dali.types as types
from cfg.config_general import cfg


class OwnInputIterator(object):
    def __init__(self, data_prep_manager):
        print("starting own input iterator")

        self.dpm = data_prep_manager

    def __call__(self):
        input_one, input_two, input_three, cur_idxs, cur_img_names = \
            self.dpm.get_next_batch()

        return input_one, input_two, input_three, cur_idxs, cur_img_names

class COCOPipeline(Pipeline):
    #See NVIDIA Dali docs
    def __init__(self, coco_captions_generator, num_threads, device_id):
        self.dpm = coco_captions_generator
        super(COCOPipeline, self).__init__(self.dpm.batch_size, num_threads, device_id, seed = 15)

        output_dtype = types.FLOAT

        #define generator
        self.generator = OwnInputIterator( self.dpm)

        #define sources
        self.input_img_e = ops.ExternalSource()
        self.input_target_class_e = ops.ExternalSource()
        self.input_target_env_e = ops.ExternalSource()
        self.input_indices_e = ops.ExternalSource()
        self.input_img_ids_e = ops.ExternalSource()


        self.resize = ops.RandomResizedCrop(
            device="gpu",
            size=self.dpm.im_size,
            random_area=[cfg.TRAIN.IMAGE_AREA_FACTOR_LOWER_BOUND, 1.0]
        )

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(self.dpm.im_size, self.dpm.im_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # imagenet mean
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],  # imagenet std
            mirror=0,
            output_dtype=output_dtype,
            output_layout=types.NCHW,
            pad_output=False)

    def define_graph(self):

        self.input_img = self.input_img_e()
        self.target_class = self.input_target_class_e()
        self.target_env = self.input_target_env_e()
        self.indices = self.input_indices_e()
        self.img_ids = self.input_img_ids_e()

        output_img = self.decode(self.input_img)
        output_img = self.resize(output_img)
        output_img = self.normalize(output_img)
        o_target_class = self.target_class.gpu()
        o_target_env = self.target_env.gpu()
        o_indices = self.indices
        o_img_ids = self.img_ids

        return output_img, o_target_class, o_target_env, o_indices, o_img_ids

    def iter_setup(self):
        (input_img, input_target_class, input_target_env, indices, img_ids) = self.generator()
        self.feed_input(self.input_img, input_img)
        self.feed_input(self.target_class, input_target_class)
        self.feed_input(self.target_env, input_target_env)
        self.feed_input(self.indices, indices)
        self.feed_input(self.img_ids, img_ids)