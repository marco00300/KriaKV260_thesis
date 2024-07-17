import tensorflow as tf

# Load the saved model from the directory 'my_saved_model'
saved_model_dir = './704samples_window_8output_classes'
model = tf.keras.models.load_model(saved_model_dir)

# Save the loaded model as 'my_model.h5' in HDF5 format
hdf5_model_path = 'my_model.h5'
model.save(hdf5_model_path, save_format='h5')
