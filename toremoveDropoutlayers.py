from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Load the original model
original_model = load_model('my_model.h5')

# Print the structure of the original model to identify the dropout layer
original_model.summary()

# Create a new input layer
inputs = Input(shape=original_model.input_shape[1:])

# Rebuild the model excluding the dropout layer
x = inputs
for layer in original_model.layers:
    if 'dropout' not in layer.name:  # Skip the dropout layers
        x = layer(x)

# Create the new model
new_model = Model(inputs=inputs, outputs=x)

# Check the structure of the new model
new_model.summary()

# Save the new model
new_model.save('my_newmodel.h5')
