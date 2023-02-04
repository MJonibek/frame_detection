# nlp701_assignment2

## Installation
```
pip install -r requirments
```

## Model
### Usage
```bash
usage: baseline.py [-h] --train TRAIN --train_label TRAIN_LABEL --test TEST
                   --pred PRED  --model MODEL_TYPE

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         path to training folder
  --train_label TRAIN_LABEL
                        path to training label file
  --test TEST           path to test folder
  --pred PRED           path to output prediction
  --model MODEL_type    model type to use. List of types: ["svm", "lr", "cnn", "lstm", "gru", "bilstm", "bigru"]. 
```

### Example

```bash
$ python src/nodel.py \
--train data/en/train-articles-subtask-2 \
--train_label data/en/train-labels-subtask-2.txt \
--test data/en/dev-articles-subtask-2 \
--pred dev-preds-subtask-2.en.txt
--model cnn
```

## Evaluation
The scorer for the subtasks is located in the ```scorers``` folder.
The official evaluation metric for the task is **micro-F1**.
The scorer also reports macro-F1.

### Usage
```bash
usage: scorer-subtask-2.py [-h] [--gold_file_path GOLD_FILE_PATH]
                           --pred_file_path PRED_FILE_PATH --frame_file_path
                           FRAME_FILE_PATH [--log_to_file]
                           [--output-for-script]

optional arguments:
  -h, --help            show this help message and exit
  --gold_file_path GOLD_FILE_PATH, -g GOLD_FILE_PATH
                        Paths to the file with gold annotations.
  --pred_file_path PRED_FILE_PATH, -p PRED_FILE_PATH
                        Path to the file with predictions
  --frame_file_path FRAME_FILE_PATH, -f FRAME_FILE_PATH
                        Path to the file with the names of the frames
  --log_to_file, -l     Set flag if you want to log the execution file. The
                        log will be appended to <pred_file_path>.log
  --output-for-script, -o
                        Prints the output in a format easy to parse for a
                        script
```

### Example
```bash
$ python scorers/scorer-subtask-2.py \
--gold_file_path data/en/dev-labels-subtask-2.txt \
--pred_file_path dev-preds-subtask-2.en.txt \
--frame_file_path scorers/frames_subtask2.txt 
```