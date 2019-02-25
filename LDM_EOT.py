import numpy as np
import warnings
import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack

_logger = utils.create_logger("LDM_EOT_attacks")

class LDM_EOT_ATTACK(Attack):
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(LDM_EOT_ATTACK, self).__init__(model, back, sess, dtypestr)

        self.feedable_kwargs = {'y': self.tf_dtype,
                                'y_target': self.tf_dtype}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate', 'perturb_window',
                                  'binary_search_steps', 'max_iterations', 'dis_metric','ensemble_size',
                                  'ground_truth', 'abort_early', 'initial_const',
                                  'clip_min', 'clip_max', 'dist_tolerance']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value (not used)
        :param clip_max: (optional float) Maximum input component value (not used)
        """
        import tensorflow as tf
        from LDM_EOT_tf import LDM_EOT_tf_ATTACK
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = LDM_EOT_tf_ATTACK(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.perturb_window, self.binary_search_steps,
                      self.max_iterations, self.dis_metric, self.ensemble_size, self.ground_truth,
                      self.abort_early, self.initial_const, self.clip_min, self.clip_max, self.dist_tolerance,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)
        wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3, perturb_window=9000,
                     binary_search_steps=5, max_iterations=1000, dis_metric=1, ensemble_size=30,
                     ground_truth = None, abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1, dist_tolerance=4500):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.perturb_window = perturb_window
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.dis_metric = dis_metric
        self.ensemble_size = ensemble_size
        self.ground_truth = ground_truth
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dist_tolerance = dist_tolerance