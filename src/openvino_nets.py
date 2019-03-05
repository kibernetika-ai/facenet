import logging


LOG = logging.getLogger(__name__)


class OpenVINONet(object):
    output_list = None

    def __init__(self, plugin, net):
        self.exec_net = plugin.load(net)
        if self.output_list:
            self.outputs = self.output_list
        else:
            self.outputs = list(iter(net.outputs))
        self.input = list(net.inputs.keys())[0]
        self.input_size = tuple(list(net.inputs.values())[0].shape[-2:])
        self.input_shape = list(net.inputs.values())[0].shape
        LOG.info(self.outputs)

    def __call__(self, img):
        output = self.exec_net.infer({self.input: img})
        out = [output[x] for x in self.outputs]
        if len(out) == 1:
            return out[0]
        else:
            return out


class RNet(OpenVINONet):
    output_list = [
        'rnet/conv5-2/conv5-2/MatMul',
        'rnet/prob1'
    ]


class ONet(OpenVINONet):
    output_list = [
        'onet/conv6-2/conv6-2/MatMul',
        'onet/conv6-3/conv6-3/MatMul',
        'onet/prob1'
    ]


class FaceDetect(OpenVINONet):
    output_list = ['detection_out']


class FaceNet(OpenVINONet):
    output_list = ['normalize']
