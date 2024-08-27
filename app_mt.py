'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import pickle  # Add import for pickle

# Remove the preprocess_fn function since it is not needed anymore

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, data):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]

    n_of_data = len(data)
    count = 0
    write_index = start
    while count < n_of_data:
        if (count + batchSize <= n_of_data):
            runSize = batchSize
        else:
            runSize = n_of_data - count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input data to input buffer '''
        for j in range(runSize):
            dataRun = inputData[0]
            dataRun[j, ...] = data[(count + j) % n_of_data].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = np.argmax((outputData[0][j]))
            write_index += 1
        count = count + runSize


def app(pickle_file, threads, model, output_file):

    with open(pickle_file, 'rb') as f:  # Load data from pickle file
        data = pickle.load(f)

    runTotal = len(data)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    '''run threads '''
    print('Starting', threads, 'threads...')
    threadAll = []
    start = 0
    for i in range(threads):
        if (i == threads - 1):
            end = len(data)
        else:
            end = start + (len(data) // threads)
        in_q = data[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, runTotal, timetotal))

    '''Save the output to a file'''
    with open(output_file, 'wb') as f:
        pickle.dump(out_q, f)
    
    print(f'Results saved to {output_file}')

    ''' post-processing '''
    '''classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    correct = 0
    wrong = 0
    print('output buffer length:', len(out_q))
    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = os.path.basename(data[i][1]).split('_', 1)  # Assuming labels are stored with data
        if (ground_truth == prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct / len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' % (correct, wrong, accuracy))
    '''
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pickle_file', type=str, default='data.pickle', help='Path to the pickle file. Default is data.pickle')
    ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model', type=str, default='model_dir/customcnn.xmodel', help='Path of xmodel. Default is model_dir/densenetx.xmodel')
    ap.add_argument('-o', '--output_file', type=str, default='y_pred', help='File to save the results. Default is results.pickle')
    
    args = ap.parse_args()

    print('Command line options:')
    print(' --pickle_file : ', args.pickle_file)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)
    print(' --output_file: ', args.output_file)

    app(args.pickle_file, args.threads, args.model, args.output_file)

if __name__ == '__main__':
    main()
