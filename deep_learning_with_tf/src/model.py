import data
import network as net
import tensorflow as tf
import time, math

from util import *


class KISNet:
    def __init__(self, model_num, model_arg, model_reuse_flag=False):
        '''
            init deep learning environment variable
        '''
        self.gpu_num = model_arg['gpu_num']
        self.model_num = model_num
        self.model_snapshot_name = model_arg['model_storage']

        self.input_layer_size = model_arg['net_input_size']
        self.output_layer_size = model_arg['net_output_size']
        self.network_type = model_arg['net_type']

        self.train_flag = False

        '''
            init deep learning hyper parameter
        '''
        self.keep_prob = model_arg['keep_prob']
        self.train_learning_rate = model_arg['learning_rate']
        self.batch_size = model_arg['batch_size']
        self.train_epoch = model_arg['epoch']
        self.LR_DECAY_OPTION = model_arg['LR_DECAY_OPTION']
        self.LR_DECAY_DROP_RATIO = model_arg['LR_DECAY_DROP_RATIO']
        self.LR_DECAY_EPOCH = model_arg['LR_DECAY_EPOCH']
        self.L2_REGULARIZATION = model_arg['L2_REGULARIZATION']
        self.L2_REGULARIZATION_SCALE = model_arg['L2_REGULARIZATION_SCALE']
        '''
            init data
        '''
        self.class_type = model_arg['class_type']
        self.mal_path = model_arg['mal_path']
        self.ben_path = model_arg['ben_path']
        self.indices = model_arg['indices']
        self.label_path = model_arg['label_path']

        '''
            init network parameter
        '''
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            tf.reset_default_graph()
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_layer_size], name='x_input')
            self.y = tf.placeholder(tf.int32, shape=[None], name='y_output')
            self.y_, self.flatten = self.inference()
            self.y_one_hot = tf.one_hot(self.y, self.output_layer_size, dtype=tf.int32, name='y_one-hot')
            self.lr = tf.placeholder(tf.float32)

            self.y_prob = tf.nn.softmax(self.y_)
            self.y_pred = tf.argmax(self.y_, 1)
            self.y_true = tf.argmax(self.y_one_hot, 1)
            self.prediction = tf.equal(self.y_pred, self.y_true)
            self.pred_cast = tf.cast(self.prediction, tf.float32)
            self.acc_cnt = tf.reduce_sum(self.pred_cast)
            self.accuracy = tf.reduce_mean(self.pred_cast)

        if not model_reuse_flag:
            self._delete_model_snapshot()

        pass

    def _delete_model_snapshot(self):
        model_storage = self.model_snapshot_name + str(self.model_num)
        if os.path.isdir(model_storage):
            print('@ delete current model (REUSE_FLAG = FALSE)')
            import shutil
            shutil.rmtree(model_storage)
        return True

    def get_model_snapshot_path(self):
        retrain_flag = True
        # create model storage
        model_storage = self.model_snapshot_name + str(self.model_num)
        if not os.path.isdir(model_storage):
            os.makedirs(model_storage)
            retrain_flag = False

        return os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(model_storage))), retrain_flag

    def inference(self):
        if self.network_type == 'ANN':
            return net.inference_ANN(self.x, self.keep_prob, self.train_flag)
        elif self.network_type == 'CNN':
            return net.inference_CNN(self.x, self.keep_prob, self.L2_REGULARIZATION_SCALE, self.train_flag)
        else:
            raise NotImplementedError
        pass

    def _set_lr(self, cur_epoch):
        if self.LR_DECAY_OPTION == 'step_decay':
            lrate = self.train_learning_rate * math.pow(self.LR_DECAY_DROP_RATIO,
                                                        math.floor(cur_epoch/self.LR_DECAY_EPOCH))
            return lrate

        elif self.LR_DECAY_OPTION == 'exp_decay':
            decayed_learning_rate = self.train_learning_rate * math.pow(self.LR_DECAY_DROP_RATIO,
                                                                         (cur_epoch/self.LR_DECAY_EPOCH))
            return decayed_learning_rate

        else:
            return self.train_learning_rate

    def train(self):
        # load data
        self.train_data = data.DataLoader(self.mal_path[self.indices[0][0]],
                                          self.ben_path[self.indices[1][0]] if self.class_type == 'binary' else list(),
                                          self.label_path, batch_size=self.batch_size, epoch=self.train_epoch,
                                          mode='train')
        self.eval_data = data.DataLoader(self.mal_path[self.indices[0][1]],
                                         self.ben_path[self.indices[1][1]] if self.class_type == 'binary' else list(),
                                         self.label_path, batch_size=self.batch_size, epoch=self.train_epoch,
                                         mode='evaluate')
        print('@ training start')
        self.train_flag = True

        # design network architecture
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            if self.L2_REGULARIZATION == 'L2':
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_, labels=self.y_one_hot))\
                       + tf.losses.get_regularization_loss()
                # optimizer: Adaptive momentum optimizer
                optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

            else:
                # loss function: softmax, cross-entropy
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_, labels=self.y_one_hot))

                # optimizer: Adaptive momentum optimizer
                optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

        # create model snapshot
        model_path, retrain_flag = self.get_model_snapshot_path()

        # training session start
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(init)

            if retrain_flag:
                model_saver.restore(sess, model_path)
                print('재학습모드')

            number_of_data = len(self.train_data)
            print('training file # : {}'.format(number_of_data))

            train_time = time.time()
            best_acc = 0.0
            for iteration, (train_data, train_label, notice) in enumerate(self.train_data):
                if not notice['signal']:
                    _cost, _, _acc = sess.run([cost, optimizer, self.accuracy],
                                              feed_dict={self.x: train_data, self.y: train_label,
                                                         self.lr: self._set_lr(notice['epoch'])})

                    if iteration and iteration % 50 == 0:
                        print('[{i} iter] cost: {cost:.4f} / acc: {acc:.4f} / elapsed time: {time:.3f}'.format(
                            i=iteration, cost=_cost, acc=_acc, time=time.time()-train_time
                        ))

                    if iteration % 100 == 0:
                        # model_saver.save(sess, model_path)
                        pass
                else:  # epoch finish
                    pass
                    temp_acc = self.evaluate_epoch_compare(sess, notice['epoch'])
                    if temp_acc > best_acc:
                        best_acc = temp_acc
                        model_saver.save(sess, model_path)
            else:
                pass
            train_time = time.time() - train_time
        print('@ training time : {}'.format(train_time))
        print('------training finish------')
        del self.train_data
        pass

    def evaluate_epoch_compare(self, sess, epoch):
        answer_cnt = 0
        number_of_data = len(self.eval_data)

        for iteration, (eval_data, eval_label) in enumerate(self.eval_data):
            try:
                acc_cnt = sess.run(self.acc_cnt, feed_dict={self.x: eval_data, self.y: eval_label})
                answer_cnt += int(acc_cnt)
            except:
                pass

        total_accuracy = float(100. * (answer_cnt / number_of_data))
        print('@ [epoch {0}] accuracy : {1:.3f}'.format(epoch, total_accuracy))
        return total_accuracy

    def evaluate(self):
        self.eval_data = data.DataLoader(self.mal_path[self.indices[0][1]],
                                         self.ben_path[self.indices[1][1]] if self.class_type == 'binary' else list(),
                                         self.label_path, batch_size=self.batch_size, epoch=self.train_epoch,
                                         mode='evaluate')
        print('@ evaluating start')
        self.train_flag = False

        # restore model snapshot
        model_path, _ = self.get_model_snapshot_path()

        # start evaluating session
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)

        actual_labels, pred_labels = list(), list()
        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            model_saver.restore(sess, model_path)

            answer_cnt = 0
            flatten_list = list()
            mal_probs = list()  # 파일에 대해 얼마나 악성이라고 생각하는지?

            number_of_data = len(self.eval_data)
            print('evaluating file # : {}'.format(number_of_data))

            eval_time = time.time()
            for iteration, (eval_data, eval_label) in enumerate(self.eval_data):
                no_eval_data = len(eval_label)
                try:
                    pred_label, actual_label, acc_cnt, flatten, y_prob = sess.run(
                        [self.y_pred, self.y_true, self.acc_cnt, self.flatten, self.y_prob],
                        feed_dict={self.x: eval_data, self.y: eval_label})
                    answer_cnt += int(acc_cnt)

                    # flatten_list += flatten.tolist()
                    mal_probs += y_prob[:, 1].tolist()

                    if iteration and iteration % 10 == 0:
                        _i = (iteration+1) * no_eval_data
                        print('[{i}/{total}] acc: {acc:.4f} / elapsed time: {time:.3f}'.format(
                            i=_i, total=number_of_data, acc=(answer_cnt/_i),
                            time=time.time()-eval_time
                        ))
                except Exception as e:
                    print(e)
                    pred_label = np.array([-1] * len(eval_label))
                    actual_label = np.array([-1] * no_eval_data)
                pred_labels += pred_label.tolist()
                actual_labels += actual_label.tolist()
            eval_time = time.time() - eval_time
        total_accuracy = float(100. * (answer_cnt / number_of_data))
        print('eval time : {0:.4f} seconds'.format(eval_time))
        print('second/file[{0}] : {1:.6f} seconds'.format(number_of_data, eval_time / number_of_data))
        print('맞은 개수: {}개'.format(answer_cnt))
        print('accuracy : {0:.3f}% [accuracy = (number_of_answer / number_of_eval_data)]'.format(total_accuracy))
        print('-----evaluating finish-----')

        # save flatten list
        # import pickle
        # with open('result/flatten{}.list'.format(self.model_num), 'wb') as f:
        #     pickle.dump(flatten_list, f)

        # save learning result
        save_result_to_csv(self.model_num, self.eval_data.get_all_file_names(), actual_labels, pred_labels,
                           mal_probs)

        # plot confusion matrix
        plot_confusion_matrix(self.model_num, actual_labels, pred_labels, self.output_layer_size)
        del self.eval_data

        return total_accuracy
