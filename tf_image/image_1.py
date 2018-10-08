import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile('image\\cat.jpg', 'rb').read()

image_data = tf.image.decode_image(image_raw_data)

with tf.Session() as sess:

    print(image_data.eval())
    print(image_data.get_shape())

    plt.imshow(image_data.eval())
    plt.show()

    # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    # encoded_image = tf.image.encode_jpeg(image_data)
    #
    # with tf.gfile.GFile('result\\cat1.jpg', 'wb') as f:
    #     f.write(encoded_image.eval())

# resize
# with tf.Session() as sess1:
#     # image_float = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
#     resized = tf.image.resize_images(image_data, size=[300, 300], method=0)
#
#     plt.imshow(resized.eval())
#     plt.show()

# crop or pad
with tf.Session() as sess2:
    croped = tf.image.resize_image_with_crop_or_pad(image_data, 1000, 1000)

    plt.imshow(croped.eval())
    plt.show()

# brightness
with tf.Session() as sess3:
    adjusted = tf.image.adjust_brightness(image_data, -0.5)

    plt.imshow(adjusted.eval())
    plt.show()

# transform
with tf.Session() as sess4:
    adjusted = tf.image.transpose_image(image_data)

    plt.imshow(adjusted.eval())
    plt.show()

# # 处理标注框
# with tf.Session() as sess5:
#     image_data = tf.image.resize_images(image_data, (180, 267), method=1)
#     batched = tf.expand_dims(tf.image.convert_image_dtype(image_data, tf.float32), 0)
#
#     boxes = tf.constant([[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]])
#     result = tf.image.draw_bounding_boxes(batched, boxes)
#
#     plt.imshow(adjusted.eval())
#     plt.show()

# 随机截图
with tf.Session() as sess6:
    boxes = tf.constant([[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]])
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_data), bounding_boxes=boxes)
    batched = tf.expand_dims(tf.image.convert_image_dtype(image_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    distort_image = tf.slice(image_data, begin, size)

    plt.imshow(image_with_box.eval())
    plt.show()