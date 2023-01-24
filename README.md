# Real_Time_Object_Detection
## Instructions to run the Project:
1. Clone the repository.
```
git clone https://github.com/GaneshLokare/Real_Time_Object_Detection.git
```
3. Create new enviroment with python==3.8

4. run command
```
python setup.py install
```
5. Download the data from provided link, unzip it and store it in the src folder.

data source: provide link

6. Install the required libraries mentioned in the requirements.txt file
``` 
pip install -r requirements.txt
```
7. run main.py

8. Go to model_training folder. 
```
cd src\model_training
```
9. Training YOLO v5 model. Run below command in the terminal.
```
python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 50
```
10. convert saved model from pytoch format to onnx format. So that we can use it with opencv.
```
python export.py --weights runs/train/Model/weights/best.pt --include onnx --simplify
```
