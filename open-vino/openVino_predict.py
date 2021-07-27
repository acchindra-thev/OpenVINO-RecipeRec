import os
import sys

import numpy as np
import logging as log
from openvino.inference_engine import IECore
from predict_helper import process_image
from datetime import datetime

def infer():
    start=datetime.now()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    cpu_extension = None
    device = 'CPU'
    input_files = [f for f in os.listdir('uploads') if os.path.isfile(os.path.join('uploads', f))]
    number_top = 1
    labels = 'labels.txt'
    predicted_ingredients = []

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = 'model.onnx'
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(input_files)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = process_image(input_files[i])
        image = image.unsqueeze_(0)
        image = image.to("cpu").float()
        images[i] = image

    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("results: ")
    if labels:
        with open(labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = [None]

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-number_top:][::-1]

        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            predicted_ingredients.append(det_label)

    print()
    print("Time for program to run is:")
    print(datetime.now()-start)
    print()
    print("Predicted ingredients:")
    print(predicted_ingredients)

    return predicted_ingredients
