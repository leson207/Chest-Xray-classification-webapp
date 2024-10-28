ENV

conda create -n cnn python=3.9 -y

conda activate cnn

pip install -r requirements.txt

git init
dvc init
dvc add artifact/data_ingestion/data
python3 setup.py install
dvc repro


Data link #link[https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone]

"https://drive.google.com/file/d/1vlhZ5c7abUKF8xXERIw6m9Te8fW7ohw3/view?usp=sharing"

"https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing"

AWS

create bucket
create iamuser, add policy
get key
install aws cli
config aws cli

https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

aws configure

create iam user to get keys

bentoml delete image_classifier_service --yes
bentoml models delete model --yes

python3 main.py

bentoml build

bentoml containerize image_classifier_service:latest -t endpoint

docker build app/client
docker compose up



/home/leson207/anaconda3/condabin:/app/bin:/app/bin:/app/bin:/usr/bin:/home/leson207/.var/app/com.visualstudio.code/data/node_modules/bin
/home/leson207/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/bin:/var/lib/flatpak/exports/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl