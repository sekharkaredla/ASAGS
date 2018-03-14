from keras.models import model_from_json
from keras.utils import vis_utils


# load model
json_file = open('model_100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load model weights
loaded_model.load_weights("model_100.h5")

vis_utils.plot_model(loaded_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("Loaded model from disk")