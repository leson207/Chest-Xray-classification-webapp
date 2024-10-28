# ENV

```bash
conda create -n cnn python=3.9 -y

conda activate cnn

pip install -r requirements.txt

git init

dvc init

python3 setup.py install
```

Data link [link](https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing)

# AWS

- create bucket
- create iamuser, add policy
- get key
- config aws cli

```bash
aws configure
```

# RUN
```bash
bentoml delete image_classifier_service --yes

bentoml models delete model --yes

dvc repro

dvc add artifact/data_ingestion/data
```

# Crate inference endpoint image
```bash
bentoml build

bentoml containerize image_classifier_service:latest -t endpoint
```
# Create client image
```bash
docker build app/client
```

# Create Compse
```bash
docker compose up
```