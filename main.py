import argparse
from config import Config
from optimizer import Optimizer
import torch
import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
# import color_analyzer
# import cv2

def getMask(img_f):

    config_file = "configs/fcn/fcn_d6_r50-d16_512x1024_40k_pws.py"
    checkpoint_file = 'pws_414_fcn/iter_16000.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    mask = inference_segmentor(model, img_f)

    return mask[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default='./input/s1.png', help="path to a directory or image to reconstruct (images in same directory should have the same resolution")

    parser.add_argument("--sharedIdentity", dest='sharedIdentity', action='store_true', help='in case input directory contains multiple images, this flag tells the optimizations that all images are for the same person ( that means the identity shape and skin reflectance is common for all images), if this flag is false, that each image belong to a different subject', required=False)
    #parser.add_argument("--no-sharedIdentity", dest='sharedIdentity', action='store_false', help='in case input directory contains multiple images, this flag tells the optimizations that all images are for the same person ( that means the identity shape and skin reflectance is common for all images), if this flag is false, that each image belong to a different subject', required=False)

    parser.add_argument("--output", required=False, default='./output/', help="path to the output directory where optimization results are saved in")
    parser.add_argument("--config", required=False, default='./optimConfig.ini', help="path to the configuration file (used to configure the optimization)")

    parser.add_argument("--checkpoint", required=False, default='', help="path to a checkpoint pickle file used to resume optimization")
    parser.add_argument("--skipStage1", dest='skipStage1', action='store_true', help='if true, the first (coarse) stage is skipped (stage1). useful if u want to resume optimization from a checkpoint', required=False)
    parser.add_argument("--skipStage2", dest='skipStage2', action='store_true', help='if true, the second stage is skipped (stage2).  useful if u want to resume optimization from a checkpoint', required=False)
    parser.add_argument("--skipStage3", dest='skipStage3', action='store_true', help='if true, the third stage is skipped (stage3).  useful if u want to resume optimization from a checkpoint', required=False)
    parser.add_argument("--skipSeg", dest='skipSeg', action='store_true')
    params = parser.parse_args()

    inputDir = params.input
    sharedIdentity = params.sharedIdentity
    outputDir = params.output + '/' + os.path.basename(inputDir.strip('/'))

    configFile = params.config
    checkpoint = params.checkpoint
    doStep1 = not params.skipStage1
    doStep2 = not params.skipStage2
    doStep3 = not params.skipStage3
    doSeg = not params.skipSeg

    config = Config()
    config.fillFromDicFile(configFile)
    if config.device == 'cuda' and torch.cuda.is_available() == False:
        print('no cuda enabled device found. switching to cpu... ')
        config.device = 'cpu'

    optimizer = Optimizer(outputDir, config)
    optimizer.run(inputDir,
                  sharedIdentity= sharedIdentity,
                  checkpoint= checkpoint,
                  doStep1= doStep1,
                  doStep2 = doStep2,
                  doStep3= doStep3,
                  doSeg = doSeg)