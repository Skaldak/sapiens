conda create -n sapiens python=3.10 -y
conda activate sapiens
conda install pytorch=2.1.0 torchvision pytorch-cuda=12.2 -c pytorch -c nvidia -y
cd _install && pip install -r requirements.txt && cd ..
cd ../libcom && pip install -e . -v && cd ../sapiens || exit
cd engine && pip install -e . -v && cd ..
cd cv && pip install -e . -v && cd ..
pip install -r cv/requirements/optional.txt
cd pretrain && pip install -e . -v && cd ..
cd pose && pip install -e . -v && cd ..
cd det && pip install -e . -v && cd ..
cd seg && pip install -e . -v && cd ..