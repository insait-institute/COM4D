micromamba create -n com4d python=3.11.13 -y

micromamba activate com4d

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

pip install -r requirements.txt

micromamba install -c conda-forge libegl libglu pyopengl -y

python -m pip install -U pip setuptools wheel
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"