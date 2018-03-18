from keras.models import model_from_json
from keras.utils.vis_utils import plot_model, model_to_dot
from IPython.display import SVG,display_svg

# load model
json_file = open('model_100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load model weights
loaded_model.load_weights("model_100.h5")

plot_model(loaded_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

display_svg(SVG(model_to_dot(loaded_model).create(prog='dot', format='svg')))
