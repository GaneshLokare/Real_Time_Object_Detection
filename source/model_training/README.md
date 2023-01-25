## Instructions to train the model:
Go to model_training folder. Run below command in the terminal.
```
cd source\model_training
```
Training YOLO v5 model. Run below command in the terminal.
```
python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 50
```
convert saved model from pytoch format to onnx format. So that we can use it with opencv. Run below command in the terminal.
```
python export.py --weights runs/train/Model/weights/best.pt --include onnx --simplify
```
