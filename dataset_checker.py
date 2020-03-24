import time
import cv2
from absl import app, flags
from absl.flags import FLAGS
import numpy as np

from modules.dataset import load_tfrecord_dataset


flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_boolean('binary_img', True, 'whether use binary file or not')
flags.DEFINE_boolean('is_ccrop', True, 'whether use central cropping or not')
flags.DEFINE_boolean('visualization', True, 'whether visualize dataset or not')


def main(_):
    if FLAGS.binary_img:
        tfrecord_name = './data/ms1m_bin.tfrecord'
    else:
        tfrecord_name = './data/ms1m.tfrecord'

    train_dataset = load_tfrecord_dataset(
        tfrecord_name, FLAGS.batch_size, binary_img=FLAGS.binary_img,
        is_ccrop=FLAGS.is_ccrop)

    num_samples = 100
    start_time = time.time()
    for idx, parsed_record in enumerate(train_dataset.take(num_samples)):
        (x_train, _), y_train = parsed_record
        print("{} x_train: {}, y_train: {}".format(
            idx, x_train.shape, y_train.shape))

        if FLAGS.visualization:
            recon_img = np.array(x_train[0].numpy() * 255, 'uint8')
            cv2.imshow('img', cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(0) == 113:
                exit()

    print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))


if __name__ == '__main__':
    app.run(main)
