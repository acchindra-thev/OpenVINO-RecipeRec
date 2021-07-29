import os
import sys
import numpy as np
import logging as log
from openvino.inference_engine import IECore
import tensorflow as tf
from datetime import datetime
import cv2

def infer():
    start=datetime.now()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    cpu_extension = None
    device = 'CPU'
    input_files = ['uploads/' + f for f in os.listdir('Uploads') if os.path.isfile(os.path.join('Uploads', f))]
    number_top = 1
    labels_img = 'labels_img.txt'
    labels_vid = 'labels_vid.txt'
    predicted_ingredients = set()

    if labels_img:
        with open(labels_img, 'r') as f:
            labels_map_img = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map_img = [None]

    if labels_vid:
        with open(labels_vid, 'r') as f:
            labels_map_vid = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map_vid = [None]

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model_img = "models/openvino/MobileNetV2/mobilenetv2.xml"
    weights_img = "models/openvino/MobileNetV2/mobilenetv2.bin"
    model_vid = "./models/OpenVINO/MobileNetV2_59/mobilenetv2_59.xml"
    weights_vid = "./models/OpenVINO/MobileNetV2_59/mobilenetv2_59.bin"
    tflite_model = "models/TFLITE/lite-model_object_detection_mobile_object_localizer_v1_1_metadata_2.tflite"

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    net_img = ie.read_network(model=model_img, weights=weights_img)
    net_vid = ie.read_network(model=model_vid, weights=weights_vid)
    net_img_exec = ie.load_network(network=net_img, device_name=device)
    net_vid_exec = ie.load_network(network=net_vid, device_name=device)
    
    # Initialize TFLITE interpreter
    interpreter = tf.lite.Interpreter(model_path = tflite_model)
    interpreter.allocate_tensors()

    #print model metadata
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    log.info("Preparing input blobs image")
    input_blob_img = next(iter(net_img.input_info))
    out_blob_img = next(iter(net_img.outputs))
    net_img.batch_size = 1
    
    log.info("Preparing input blobs video")
    input_blob_vid = next(iter(net_vid.input_info))
    out_blob_vid = next(iter(net_vid.outputs))
    net_vid.batch_size = 1
    
    # Open Video Capture
    cap = cv2.VideoCapture(input_files[0])
    if cap.isOpened() == False:
        print("Error opening video file")
    print(input_files[0].split('.')[-1])
    if input_files[0].split('.')[-1] in ['jpeg', 'jpg', 'png']:
        log.info("Starting inference in synchronous mode")
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            frame = np.transpose(frame, [2,0,1]) / 255
            frame = np.reshape(frame, (1, 3, 224, 224))
            res = net_img_exec.infer(inputs={input_blob_img: frame})

            # Processing output blob
            log.info("Processing output blob")
            res = res[out_blob_img]

            log.info("results: ")
            for i, probs in enumerate(res):
                probs = np.squeeze(probs) #[np.squeeze(probs) > .5]
                top_ind = np.argsort(probs)[-number_top:][::-1]

                for id in top_ind:
                    det_label = labels_map_img[id] if labels_map_img else "{}".format(id)
                    predicted_ingredients.add(det_label)
                    print(det_label)
    else:
        fcount = 0
        while cap.isOpened():
            # Capture each frame - slower than native openvino inference
            ret, frame = cap.read()
            fcount += 1
            if (ret == True) and ((fcount % 10) == 0):
                print(f'frame: {fcount}')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped = frame[:, 420:1500] # 1080x1080
                small = cv2.resize(cropped, (192,192), interpolation = cv2.INTER_AREA)
                small = np.resize(small, (1, 192, 192, 3))
                interpreter.set_tensor(input_details[0]['index'], small)
                interpreter.invoke()
                out_dict = {
                    'detection_boxes' : interpreter.get_tensor(output_details[0]['index']),
                    'detection_scores' : interpreter.get_tensor(output_details[2]['index'])}
                out_dict['detection_boxes'] = out_dict['detection_boxes'][0][:number_top]
                out_dict['detection_scores'] = out_dict['detection_scores'][0][:number_top]
                for i, score in enumerate(out_dict['detection_scores']):
                    if score > .5:
                        ymin, xmin, ymax, xmax = (out_dict['detection_boxes'][i]*1080).astype(int)
                        roi = cropped[ymin:ymax, xmin:xmax]
                        if roi.shape[0] < 80 or roi.shape[1] < 80:
                            continue
                        roi = cv2.resize(roi, (224, 224), interpolation = cv2.INTER_AREA)
                        #cv2.imshow('test', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                        #cv2.waitKey(500)
                        #cv2.destroyAllWindows()
                        roi = (np.transpose(roi, [2,0,1]).astype(float) / 127.5) -1
                        roi = np.reshape(roi, (1, 3, 224, 224))
                        # Start sync inference
                        res = net_vid_exec.infer(inputs={input_blob_vid: roi})

                        # Processing output blob
                        res = res[out_blob_vid]

                        for i, probs in enumerate(res):
                            probs = np.squeeze(probs) #[np.squeeze(probs) > .5]
                            top_ind = np.argsort(probs)[-number_top:][::-1]
                            if probs[top_ind] < .6:
                                continue
                            for id in top_ind:
                                det_label = labels_map_vid[id] if labels_map_vid else "{}".format(id)
                                predicted_ingredients.add(det_label)
                                print(det_label)
            elif (ret == False):
                cap.release()

    print()
    print("Time for program to run is:")
    print(datetime.now()-start)
    print()
    print("Predicted ingredients:")
    print(predicted_ingredients)

    return predicted_ingredients
