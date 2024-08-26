import tensorflow as tf
from tensorflow.keras import layers, models

# Load the existing model
model = tf.keras.models.load_model('my_newmodel.h5')

# Display the original model summary
print("Original model summary:")
model.summary()

# Define the new shape for the Reshape layer
new_shape = (32, )  # Adjust this shape according to your needs

# Extract the input tensor from the existing model
input_tensor = model.input

# Manually apply layers up to and including 'dense_0'
x = model.get_layer(name='conv_0')(input_tensor)  # Example for the first layer
x = model.get_layer(name='batchnorm_0')(x)
x = model.get_layer(name='Mpool_0')(x)
x = model.get_layer(name='conv_1')(x)
x = model.get_layer(name='batchnorm_1')(x)
x = model.get_layer(name='Mpool_1')(x)
x = model.get_layer(name='flatten_0')(x)

# Apply 'dense_0' and then the Reshape layer
x = model.get_layer(name='dense_0')(x)
x = layers.Reshape(new_shape)(x)

# Apply layers after the Reshape layer
x = model.get_layer(name='batchnorm_2')(x)
x = model.get_layer(name='dense_1')(x)
x = model.get_layer(name='batchnorm_3')(x)
x = model.get_layer(name='OUT_dense_2')(x)

# Create the new model with the updated architecture
new_model = models.Model(inputs=input_tensor, outputs=x)

# Display the modified model summary
print("Modified model summary:")
new_model.summary()
new_model.save('my_reshaped_model.h5') 