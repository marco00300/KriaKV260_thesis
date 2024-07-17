# KriaKV260_thesis
I'm trying to quantize and compile a tf model for the Kria KV260.

Workflow
1-I converted the model in a HDF5 file

2-I used my_quantize_fn.py to quantize my_model.h5 but I encountered some issues: 
  raise ValueError("Shapes %s and %s are incompatible" % (self, other))
  ValueError: Shapes (4, 32) and (268, 32) are incompatible
  In the network I found that probably the error was due to Vitis.AI (1.4) incompatibility with dropout layers.
  
3-With toremoveDropoutlayers.py I removed the dropout layers thus I created my_newmodel.hf which doesn't have dropout layers.

4-Again I used my_quantize_fn.py to quantize my_newmodel.h5 but I encountered some issues: 
  ValueError: Input 0 of layer batchnorm_2 is incompatible with the layer: : expected min_ndim=4, found ndim=2. 
  Full shape         received: [None, 32]


