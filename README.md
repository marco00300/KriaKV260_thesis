# KriaKV260_thesis
I'm trying to quantize and compile a tf model for the Kria KV260 with Vitis AI.

convertertoH5.py : to convert the model to .h5 format;

toremoveDropoutlayers.py: normally the quantizer delete automatically the dropout layers, in my case I had some issues with this so I created a file to delete manually each dropout layer;

my_quantize_fn.py: file to quantize the .h5 model;

compile.sh: script to compile the model;

app_mt.py: to run the code on the target board;

classification_report.py: to get some evaluation metrics;


