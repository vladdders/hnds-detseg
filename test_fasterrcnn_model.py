import os
import sys
import torch
import torchvision
import cv2.cv2 as cv2
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.backends.cudnn as cudnn_backend

from webcam import Webcam
from downloader import download_file_from_google_drive


SOURCE = "https://docs.google.com/uc?export=download&id=1gNIBp30qNFcXM5u-BGI2rurdhLa8Naxs"
MODEL_PATH = "./models/frcnn_hands.pth"
NUMBER_OF_CLASSES = 2 # hand/no hand
cudnn_backend.CudnnModule.benchmark = True
cudnn_backend.CudnnModule.enabled = True


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_prediction(pred, threshold):
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t):
        boxes = [pred[0]['boxes'][i].squeeze().detach().cpu().numpy() for i in pred_t]
        return boxes
    return []


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
        model.load_state_dict(torch.load('./models/frcnn_hands.pth', map_location=device))
        model.to(device)
        model.eval()

        if recording:
            movie = cv2.VideoWriter(f'./recordings/hand_frcnn_{camera_type}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 8, (640, 480))

        with torch.no_grad():

            while collector.started:

                image, _ = collector.read()

                if image is not None:

                    orig = image.copy()

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = transforms.ToTensor()(image).to(device)

                    out = model([image])

                    boxes = get_prediction(pred=out, threshold=.7)

                    try:

                        for box in boxes:
                            cv2.rectangle(img=orig,
                                          pt1=(box[0], box[1]),
                                          pt2=(box[2], box[3]),
                                          color=(0, 255, 255),
                                          thickness=2)

                        if recording:
                            movie.write(orig)

                        cv2.imshow("mask", orig)
                        k = cv2.waitKey(1)

                        if k == ord('q'):
                            collector.stop()

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