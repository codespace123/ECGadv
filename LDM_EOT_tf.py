from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import time
import cleverhans.utils as utils
import cleverhans.utils_tf as utils_tf
import itertools
from mysoftdtw_c_wd import mysoftdtw

_logger = utils.create_logger("myattacks.tf")

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')
data_len = 9000

def ZERO():
    return np.asarray(0., dtype=np_dtype)


def EOT_time(x, start, ensemble_size):
    def randomizing_EOT(x, start):
        rand_i = tf.expand_dims(tf.random_uniform((), start+1, data_len+1, dtype=tf.int32), axis=0)
        p = tf.concat([rand_i, data_len - rand_i], axis=0)
        x1, x2 = tf.split(x, p, axis=1)
        res = tf.reshape(tf.concat([x2, x1], axis=1), [1, data_len, 1])
        return res

    return tf.concat([randomizing_EOT(x, start) for _ in range(ensemble_size)], axis=0)

def Seq1():
   tmp = np.zeros((1, 9001, 1), dtype=np_dtype)
   tmp[:,1:9001,:] = 1.
   return np.asarray(tmp, dtype=np_dtype)

def zero_mean(batch_newdata):
    data_mean, data_var = tf.nn.moments(batch_newdata, axes=1)
    mean = tf.expand_dims(tf.tile(data_mean, [1, data_len]), 2)
    var = tf.expand_dims(tf.tile(data_var, [1, data_len]), 2)
    return (batch_newdata - mean) / tf.sqrt(var)

class LDM_EOT_tf_ATTACK(object):

    def __init__(self, sess, model, batch_size, confidence,
                 targeted, learning_rate, perturb_window,
                 binary_search_steps, max_iterations, dis_metric, ensemble_size,
                 ground_truth, abort_early, initial_const,
                 clip_min, clip_max, dist_tolerance, num_labels, shape):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
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
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        :param dis_metric: the distance metirc, 1 for l2, 2 for smoothness
        :param ensemble_size: the ensemble size for EOT_time
        :param perturb_window: windows size
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.perturb_window = perturb_window
        self.MAX_ITERATIONS = max_iterations
        self.dis_metric = dis_metric
        self.ensemble_size = ensemble_size
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.dist_tolerance = dist_tolerance
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.perturb_window = perturb_window
        self.repeat = binary_search_steps >= 10
        self.ground_truth = ground_truth
        self.shape = shape = tuple([batch_size] + list(shape))
        shape_perturb = tuple([batch_size, perturb_window, 1])

        #  self.transform_shape = transform_shape = tuple([transform_batch_size] + list(transform_shape))
        #        self.shape = shape = tuple(list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape_perturb, dtype=np_dtype))




        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype,
                                name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf_dtype, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf_dtype,
                                 name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape,
                                          name='assign_timg')
        self.assign_tlab = tf.placeholder(tf_dtype, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf_dtype, [batch_size],
                                           name='assign_const')


        pad_zero = tf.constant([[0,0], [0, data_len - perturb_window], [0,0]])
        modifier_tile = tf.reshape(tf.pad(modifier, pad_zero, "CONSTANT"), [1, data_len, 1])

        if perturb_window == data_len:
            start_p = tf.constant(0)
        else:
            start_p = perturb_window

        self.newimg = tf.slice(modifier_tile, (0, 0, 0), shape) + self.timg
        batch_newdata = EOT_time(modifier_tile, start_p, self.ensemble_size) + self.timg

        self.batch_newimg = zero_mean(batch_newdata)

        self.loss_batch = model.get_logits(self.batch_newimg)

        self.batch_tlab = tf.tile(self.tlab, (self.batch_newimg.shape[0], 1))

        self.xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.loss_batch, labels=self.batch_tlab))

        #self.xent_rest = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.loss_batch_rest, labels=self.batch_tlab_rest))
        self.outputxent = tf.expand_dims(self.xent, 0)
        self.output = tf.expand_dims(tf.reduce_mean(self.loss_batch, axis=0), 0)

        # distance to the input data
        #        self.other = (tf.tanh(self.timg) + 1) / \
        #            2 * (clip_max - clip_min) + clip_min
        #        self.l2dist = reduce_sum(tf.square(self.newimg - self.other),
        #                                 list(range(1, len(shape))))


        if self.dis_metric == 1:
            self.dist = tf.reduce_sum(tf.square(modifier_tile), list(range(1, len(shape))))
        else:
            if self.dis_metric == 2:
                _, distvar = tf.nn.moments(
                    tf.multiply(tf.concat([modifier_tile, [[[0.]]]], 1) - tf.concat([[[[0.]]], modifier_tile], 1), Seq1()),
                    axes=[1])
                self.dist = 10000 * distvar


        # sum up the losses
        self.loss2 = tf.reduce_sum(self.dist)
        self.loss1 = tf.reduce_sum(self.const *self.xent)
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])



        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(("Running attack on instance " +
                           "{} of {}").format(i, len(imgs)))
            r.extend(self.attack_batch(imgs[i:i + self.batch_size],
                                       targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y
        batch_size = self.batch_size

        #        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        #        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        #        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        #        imgs = (imgs * 2) - 1
        # convert to tanh-space
        #        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e6] * batch_size
        o_bestscore = [-1] * batch_size
        #        o_bestattack = np.copy(oimgs)
        o_bestattack = np.copy(imgs)
        o_bestConst = [-1] * batch_size
        o_bestdist = [-1] * batch_size
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
            bestl2 = [1e6] * batch_size
            bestdist = [-1] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step {} of {}".
                          format(outer_step, self.BINARY_SEARCH_STEPS))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            print("current const:", CONST)
            prev = 1e9
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg, xent, loss_batch = self.sess.run([self.train,
                                                         self.loss,
                                                         self.dist,
                                                         self.output,
                                                         self.newimg,
                                                         self.outputxent,
                                                         self.loss_batch])


                print(
                    'Iteration {} of {}: loss={:.3g} " + "dis={:.3g} xent={:.3g}'.format(iteration, self.MAX_ITERATIONS, l,
                                                                                     np.mean(l2s), np.mean(xent)))
                print('logits:', scores)
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                        iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l
                # adjust the best result found so far
                for e, (l2, sc, ii, dist, xe) in enumerate(zip(itertools.repeat(l, len(scores)), scores, nimg, l2s, xent)):
                    dist = dist/self.perturb_window
                    lab = np.argmax(batchlab[e])
                    if xe < bestl2[e] and compare(sc, lab):
                        bestl2[e] = xe
                        bestscore[e] = np.argmax(sc)
                        bestdist[e] = dist
                    if xe < o_bestl2[e] and compare(sc, lab) and (dist > 0.5 and dist < 1.5):
                        o_bestl2[e] = xe
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_bestConst[e] = CONST[e]
                        o_bestdist[e] = dist
            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                        bestscore[e] != -1 and bestdist[e] > 1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            _logger.debug("  Successfully generated adversarial examples " +
                          "on {} of {} instances.".
                          format(sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        print(o_bestscore)
        print("best distance:", o_bestdist)
        print("best c:",o_bestConst)
        print("best xent:",o_bestl2)
        return o_bestattack

# ---------------------------------------------------------------------------------

