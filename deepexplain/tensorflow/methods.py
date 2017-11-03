from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops, gen_math_ops, math_ops
from collections import OrderedDict


def np_activation(name):
    f = None
    if 'Relu' in name:
        f = lambda x: x * (x > 0)
    elif 'Sigmoid' in name:
        f = lambda x: 1 / (1 + np.exp(-x))
    elif 'Tanh' in name:
        f = np.tanh
    elif 'Softplus' in name:
        f = lambda x: np.log(1 + np.exp(x))
    return f


def activation(name):
    f = None
    if 'Relu' in name:
        f = tf.nn.relu
    elif 'Sigmoid' in name:
        f = tf.nn.sigmoid
    elif 'Tanh' in name:
        f = tf.nn.tanh
    elif 'Softplus' in name:
        f = tf.nn.softplus
    return f


def grad_activation(name):
    f = None
    if 'Relu' in name:
        f = lambda x: tf.where(tf.greater_equal(x, 0), tf.ones_like(x), tf.zeros_like(x))
    elif 'Sigmoid' in name:
        f = lambda x: tf.exp(x) / tf.square(tf.exp(x) + 1.0)
    elif 'Tanh' in name:
        f = lambda x: 1.0 / tf.square(tf.cosh(x))
    elif 'Softplus' in name:
        f = lambda x:  tf.exp(x) / (tf.exp(x) + 1.0)
    return f



class GradientBasedMethod(object):
    def __init__(self, T, X, xs, session):
        self.T = T
        self.X = X
        self.xs = xs
        self.session = session

    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)[0]

    def run(self):
        attributions = self.get_symbolic_attribution()
        return self.session.run(attributions, {self.X: self.xs})

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return grad * grad_activation(op.name)(op.inputs[0])


class DummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)[0]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)

"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""
class Saliency(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return tf.abs(tf.gradients(self.T, self.X)[0])

"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


class GradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)[0] * self.X


"""
Integrated Gradients
https://arxiv.org/pdf/1703.01365.pdf
"""


class IntegratedGradients(GradientBasedMethod):

    def __init__(self, T, X, xs, session, steps=100, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, xs, session)
        self.steps = steps
        self.baseline = baseline

    def run(self):
        if self.baseline is None: self.baseline = np.zeros_like(self.xs)
        else: assert self.baseline.shape == self.xs.shape, 'Baseline must have the same shape of input'
        attributions = self.get_symbolic_attribution()
        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = self.xs * alpha
            _attr = self.session.run(attributions, {self.X: xs_mod})
            if gradient is None: gradient = _attr
            else: gradient += _attr
        return gradient * (self.xs - self.baseline) / self.steps


"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""


class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, xs, session, epsilon=1e-4):
        super(EpsilonLRP, self).__init__(T, X, xs, session)
        self.eps = epsilon

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)[0] * self.X

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + cls.eps *
                                tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))



attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'e-lrp': (EpsilonLRP, 4),
})
_ENABLED_METHOD_CLASS = None

@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS
    if _ENABLED_METHOD_CLASS is not None:
        return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return grad * grad_activation(op.name)(op.inputs[0])


class DeepExplain(object):

    def __init__(self, graph=tf.get_default_graph(), sess=tf.get_default_session()):
        self.method = None
        self.batch_size = None
        self.graph = graph
        self.session = sess
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())


    def get_override_map(self):
        return {'Relu': 'DeepExplainGrad',
                'Sigmoid': 'DeepExplainGrad',
                'Tanh': 'DeepExplainGrad',
                'Softplus': 'DeepExplainGrad'}

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)

    def explain(self, method, T, X, xs, **kwargs):
        global _ENABLED_METHOD_CLASS
        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        print('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))

        method = method_class(T, X, xs, self.session, **kwargs)
        _ENABLED_METHOD_CLASS = method
        result = method.run()
        _ENABLED_METHOD_CLASS = None
        return result


# class BaseAttributionMethod(object):
#     def __init__(self, batch_size=1, input_batch_size=None):
#         self.model = None
#         self.batch_size = batch_size
#         self.input_batch_size = batch_size
#         self.name = 'BaseClass'
#         self.eval_x = None
#         self.eval_y = None
#         self.target_input_idx = None
#         self.target_output_idx = None
#
#     """
#     Bind model. If target input and output are not first and last layers
#     of the network, target layer indeces should also be provided.
#     """
#     def bind_model(self, model):
#         self.model = model
#         print ('Target output: %s' % self.model.layers[-1].output)
#
#     def gradient_override_map(self):
#         return {}
#
#     def sanity_check(self, eval_x, eval_y, maps):
#         pass
#
#     def target_input(self):
#         return self.model.layers[0].input
#
#     def target_output(self):
#         return self.model.layers[-1].output
#
#     def get_numeric_sensitivity(self, eval_x, eval_y, shape=None, **kwargs):
#         """
#         Return sensitivity map as numpy array
#         :param eval_x: numpy input to the model
#         :param eval_y: numpy labels
#         :param shape: if provided, the sensitivity will be reshaped to this shape
#         :return: numpy array with shape of eval_x
#         """
#         self.eval_x = eval_x
#         self.eval_y = eval_y
#         sensitivity = None
#         shape = K.int_shape(self.target_input())[1:]
#
#         #y_ = Input(batch_shape=(self.batch_size,) + tuple(eval_y.shape[1:]))  # placeholder for labels
#         print ('eval_y shape: ', eval_y.shape)
#         y_ = K.placeholder(shape=(None, ) + tuple(eval_y.shape[1:]), name='y_label')  # placeholder for labels
#         symbolic_sensitivity = self.get_symbolic_sensitivity(y_)
#         evaluate = K.function([self.model.inputs[0], y_], [symbolic_sensitivity])
#
#         for i in range(int(len(eval_x) / self.batch_size) + 1):
#             x = eval_x[self.batch_size*i:self.batch_size*(i+1)]
#             y = eval_y[self.batch_size*i:self.batch_size*(i+1)]
#             self.input_batch_size = len(x)
#             if self.input_batch_size > 0:
#                 tmp = evaluate([x, y])[0]
#                 if sensitivity is None:
#                     sensitivity = tmp
#                 else:
#                     sensitivity = np.append(sensitivity, tmp, axis=0)
#
#         if shape is not None:
#             sensitivity = sensitivity.reshape((self.eval_x.shape[0],) + shape)
#
#         # Run sanity check if available
#         self.sanity_check(eval_x, eval_y, sensitivity)
#         return sensitivity
#
#     def get_symbolic_sensitivity(self, eval_y):
#         """
#         Return sensitivity map as numpy array
#         :param eval_y: placeholder (Tensor) for labels
#         :return: Tensor with shape of eval_x
#         """
#         raise RuntimeError('Calling get_symbolic_sensitivity on abstract class')



