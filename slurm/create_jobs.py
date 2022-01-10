import os

from glob import glob

names = ['VGG19', 'ResNet18NoSkip', 'ResNet50', 'ResNet50NoSkip']
job_name = 'eval_vgg_and_noskips'

with_slurm = False

os.system('mkdir -p jobs')
file = f'jobs/volume/ensemble_volumes_{job_name}.txt'

all_model_dirs = []
for name in names:
    all_model_dirs += glob(f'outputs/{name}/*')

string = "python basin_volume.py --model_dir {}\n"

if with_slurm:
    string = f'sbatch slurm/slurm.sh {string}'

with open(file, 'w') as f:
    for model_dir in all_model_dirs:
        f.write(string.format(model_dir))

print(f'Written to file {file}')

os.system(f'cat {file}')
