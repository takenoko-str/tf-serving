from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from PIL import Image
import numpy as np

tf.app.flags.DEFINE_string('server', 'localhost:9000', '')
tf.app.flags.DEFINE_string('image', './beagle.png', '')
FLAGS = tf.app.flags.FLAGS


def main():

    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    img = Image.open(FLAGS.image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width_height = (224, 224)
    img = img.resize(width_height)

    image_data = np.asarray(img, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)
    image_data.reshape((1,) + image_data.shape)

    image_data = image_data / 255.

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "imagenet"
    request.model_spec.signature_name = "predict"
    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, dtype="float32", shape=[1, 224, 224, 3]))

    result = stub.Predict(request, 5.0)
    print(result)

if __name__ == '__main__':
    main()

