import os
import sys
import cv2
import numpy
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.backends.cudnn as cudnn_backend

from webcam import Webcam
from downloader import download_file_from_google_drive


SOURCE = 'https://docs.google.com/uc?export=download&id=1IeS9o_xfB21LieFSSa8ROlap4hhivdaz'
MODEL_PATH = "./models/mrcnn_hands.pth"
NUMBER_OF_CLASSES = 2 # hand/no hand
cudnn_backend.CudnnModule.enabled = True
cudnn_backend.CudnnModule.benchmark = True


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_prediction(pred, threshold):
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t):
        pred_t = pred_t[-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        masks = masks[:pred_t + 1]
        return masks
    return []


def get_bw_mask(mask):
    if len(mask.shape) > 1:
        r = numpy.zeros_like(mask).astype(numpy.uint8)
        g = numpy.zeros_like(mask).astype(numpy.uint8)
        b = numpy.zeros_like(mask).astype(numpy.uint8)

        r[mask == 1], g[mask == 1], b[mask == 1] = [1, 1, 1]
        mask = numpy.stack([r, g, b], axis=2)
        return mask


def main():
    try:

        camera_type = sys.argv[1]
        recording = False

        if len(sys.argv) == 3:
            if sys.argv[2] == "record":
                recording = True
            else:
                recording = False

        if camera_type == "webcam":

            collector = Webcam(video_width=640, video_height=480)
            collector.start()

        else:
            print("No such camera {camera_type}")
            collector = None
            exit(-1)

        if not os.path.isfile(MODEL_PATH):
            print("Downloading model, please wait...")
            download_file_from_google_drive(SOURCE, MODEL_PATH)
            print("Done downloading the model.")

        # get device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # initialise model
        model = get_model_instance_segmentation(NUMBER_OF_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        if recording:
            movie = cv2.VideoWriter(f'./recordings/hand_maskrcnn_{camera_type}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 8, (640, 480))

        with torch.no_grad():

            while collector.started:

                image, _ = collector.read()

                if image is not None:

                    orig = image.copy()

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = transforms.ToTensor()(image).to(device)

                    out = model([image])
                    masks = get_prediction(pred=out, threshold=.5)

                    try:

                        bw_masks = list(map(get_bw_mask, masks))

                        total_mask = sum(bw_masks)
                        total_mask[total_mask > 1] = 1

                        img = numpy.multiply(orig, total_mask)

                        if recording:
                            movie.write(img)
                        cv2.imshow("mask", img)
                        k = cv2.waitKey(1)

                        if k == ord('q'):
                            collector.stop()

                    except TypeError:
                        pass

                    except Exception as e:
                        print(e)

    finally:
        print("Stopping stream.")
        collector.stop()
        if recording:
            movie.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
