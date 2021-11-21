wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip -d . maestro-v3.0.0.zip
cat packages.txt | xargs sudo apt-get install -y
pip3 install -r requirements.txt
