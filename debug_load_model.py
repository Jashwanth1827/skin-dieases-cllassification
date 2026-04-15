import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.mixed_precision import Policy

def custom_input_layer(**config):
    config.pop('batch_shape', None)
    config.pop('batch_input_shape', None)
    config.pop('ragged', None)
    config.pop('sparse', None)
    return InputLayer(**config)

def custom_dtype_policy(**config):
    return Policy('float32')

try:
    print('trying load')
    model = load_model('skin_disease_model.h5', compile=False, custom_objects={'InputLayer':custom_input_layer, 'DTypePolicy':custom_dtype_policy})
    print('loaded', model)
except Exception as e:
    traceback.print_exc()
