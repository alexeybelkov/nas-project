# Train-Free NAS for Tabular Data     
## [Neural architecture search for tabular DNN via train-free NAS](https://github.com/diff7/Efficient-DL-models-Seminars?tab=readme-ov-file#3-neural-architecture-search-for-tabular-dnn-via-train-free-nas)        

> [!WARNING] 
> All further actions will be carried out in project folder, so clone this repo and cd into it     

```shell
git clone https://github.com/alexeybelkov/nas-project.git
cd nas_project
```

### Environment
If you want to mimic our environment, then you can use conda:
```shell
conda create -n nas_project python=3.10.12
conda activate nas_project
pip install -r requirements.txt
```

### Run

To run experiments the one needs to run the following commands:
```shell
python3 run.py --config=config.json --device=cuda
```
You may provide your own config file in json format

### Experiments

Detailed description of experiments can be found in the *REPORT.pdf*

By default *run.py* will run experiments as in the *REPORT.pdf* with default parameters from config.json file        
Expected output will be:
- Mean and std of MSE ([Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)) based on *num_launches*
- Number of parameters
- Model     

It will be printed out during *run.py *





