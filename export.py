"""Export a YOLOv5 *.pt model to TorchScript, ONNX, CoreML formats

Usage:
    $ python path/to/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.common import NMS, NMS_Export
from models.common import Conv
from models.yolo import Detect
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device
from utils.proto.pytorch2proto import prepare_model_for_layer_outputs, retrieve_onnx_names

from utils.proto import tidl_meta_arch_yolov5_pb2
from google.protobuf import text_format


def export_torchscript(model, img, file, optimize):
    # TorchScript model export
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')
        ts = torch.jit.trace(model, img, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ts
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, img, file, opset, train, dynamic, simplify, output_names=None):
    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        check_requirements(('onnx', 'onnx-simplifier'))
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'] if output_names is None else output_names,
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"{prefix} run --dynamic ONNX model inference with detect.py: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_prototxt(model, img, file, simple_search):
    # Prototxt export for a given ONNX model
    prefix = colorstr('Prototxt:')
    onnx_model_name = str(file.with_suffix('.onnx'))

    for module in model.modules():
        if isinstance(module, Detect):
            anchor_grid = torch.squeeze(module.anchor_grid)
            break
    num_heads = anchor_grid.shape[0]
    matched_names = retrieve_onnx_names(img, model, onnx_model_name, simple_search=simple_search)
    prototxt_name = onnx_model_name.replace('onnx', 'prototxt')

    background_label_id = -1
    num_classes = model.nc
    assert len(matched_names) == num_heads; "There must be a matched name for each head"
    proto_names = [f'{matched_names[i]}' for i in range(num_heads)]
    yolo_params = []
    for head_id in range(num_heads):
        yolo_param = tidl_meta_arch_yolov5_pb2.TIDLYoloParams(input=proto_names[head_id],
                                                        anchor_width=anchor_grid[head_id,:,0],
                                                        anchor_height=anchor_grid[head_id,:,1])
        yolo_params.append(yolo_param)

    nms_param = tidl_meta_arch_yolov5_pb2.TIDLNmsParam(nms_threshold=0.65, top_k=30000)
    detection_output_param = tidl_meta_arch_yolov5_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_yolov5_pb2.CODE_TYPE_YOLO_V5, keep_top_k=300,
                                            confidence_threshold=0.005)

    yolov3 = tidl_meta_arch_yolov5_pb2.TidlYoloOd(name='yolo_v3', output=["detections"],
                                            in_width=img.shape[3], in_height=img.shape[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            )
    arch = tidl_meta_arch_yolov5_pb2.TIDLMetaArch(name='yolo_v3', tidl_yolo=[yolov3])

    with open(prototxt_name, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


def export_coreml(model, img, file):
    # CoreML model export
    prefix = colorstr('CoreML:')
    try:
        import coremltools as ct

        print(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')
        model.train()  # CoreML exports should be placed in model.train() mode
        ts = torch.jit.trace(model, img, strict=False)  # TorchScript model
        model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        model.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


def run(weights='./yolov5s.pt',  # weights path
        img_size=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx', 'coreml'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=False,  # ONNX: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        export_nms=False,
        simple_search=False
        ):
    t = time.time()
    include = [x.lower() for x in include]
    img_size *= 2 if len(img_size) == 1 else 1  # expand
    file = Path(weights)

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = model.names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(img)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    if export_nms:
        nms = NMS(conf=0.001)
        nms_export = NMS_Export(conf=0.001)
        #y_export = nms_export(y)
        #y = nms(y)
        #assert (torch.sum(torch.abs(y_export[0]-y[0]))<1e-6)
        model_nms = torch.nn.Sequential(model, nms_export)
        model_nms.train() if train else model_nms.eval()
        output_names = ['detections']

    # Exports
    if 'torchscript' in include:
        export_torchscript(model, img, file, optimize)
    if 'onnx' in include:
        if export_nms:
            export_onnx(model_nms, img, file, opset, train, dynamic, simplify, output_names)
        else:
            export_onnx(model, img, file, opset, train, dynamic, simplify)

        export_prototxt(model, img, file, simple_search)

    if 'coreml' in include:
        export_coreml(model, img, file)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--export-nms', action='store_true', help='export the nms part in ONNX model')  # ONNX-only, #opt.grid has to be set True for nms export to work
    opt = parser.parse_args()
    return opt


def main(opt):
    set_logging()
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
