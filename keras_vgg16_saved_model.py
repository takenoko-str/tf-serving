from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
import keras.backend as K
from keras.applications import vgg16
from keras.models import Model

MODEL_VERSION = 4
K.set_learning_phase(0)


model = vgg16.VGG16(include_top=True, weights='imagenet')

config = model.get_config()
weights = model.get_weights()
new_model = Model.from_config(config)
new_model.set_weights(weights)

export_path = '/tmp/model/' + str(MODEL_VERSION)
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'images': new_model.input},
                                  outputs={'scores': new_model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()
    print ('Successfully exported model to {}'.format(FLAGS.output_dir))

