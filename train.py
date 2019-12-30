from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import load_yaml, get_ckpt_inf
import modules.dataset as dataset


flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True)
    model.summary(line_length=80)

    if cfg['train_dataset']:
        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        dataset_len = 1
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])

    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print('[*] load ckpt from {}'.format(ckpt_path))
        epochs, steps = get_ckpt_inf(ckpt_path)
        model.load_weights(ckpt_path)
    else:
        print('[*] training from scratch.')
        epochs = 1
        steps = 0

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        while epochs < cfg['epochs'] + 1:
            for inputs, labels in train_dataset:
                with tf.GradientTape() as tape:
                    logist = model(inputs, training=True)
                    reg_loss = tf.reduce_sum(model.losses)
                    pred_loss = loss_fn(labels, logist)
                    total_loss = pred_loss + reg_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                if steps % 5 == 0 and steps > 0:
                    verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
                    print(verb_str.format(epochs, cfg['epochs'],
                                          steps % steps_per_epoch,
                                          steps_per_epoch,
                                          total_loss.numpy(),
                                          learning_rate.numpy()))

                    with summary_writer.as_default():
                        tf.summary.scalar(
                            'loss/total loss', total_loss, step=steps)
                        tf.summary.scalar(
                            'loss/pred loss', pred_loss, step=steps)
                        tf.summary.scalar(
                            'loss/reg loss', reg_loss, step=steps)
                        tf.summary.scalar(
                            'learning rate', optimizer.lr, step=steps)

                if steps % 1000 == 0 and steps > 0:
                    print('[*] save ckpt file!')
                    ckpt_name = 'checkpoints/{}/e_{}_s_{}.ckpt'
                    model.save_weights(
                        ckpt_name.format(cfg['sub_name'], epochs, steps))

                steps += 1
            epochs += 1
    else:
        model.compile(optimizer=optimizer, loss=loss_fn,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        mc_callback = ModelCheckpoint(
                'checkpoints/' + cfg['sub_name'] + '/e_{epoch}_s_{batch}.ckpt',
                save_freq=1000 * cfg['batch_size'] + 1, verbose=1,
                save_weights_only=True)
        tb_callback = TensorBoard(log_dir='logs/',
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        callbacks = [mc_callback, tb_callback]

        history = model.fit(train_dataset,
                            epochs=cfg['epochs'],
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            initial_epoch=epochs - 1)


if __name__ == '__main__':
    app.run(main)
