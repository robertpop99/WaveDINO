"""
Generates the SLURM job scripts for every experiment in the paper, plus one submit script per experiment group:

    submit_all.sh             train every model on both sites (Tables I-III, incl. the ablation variants)
    submit_all_checkpoint.sh  continue training from the newest checkpoint (--load-checkpoint)
    submit_all_test.sh        synthetic test (Table I, Table II top) and real-data GNSS evaluation with full-frame
                              sliding window + resize (Table II bottom, Supp. S5)
    submit_all_synth.sh       synthetic analyses with analysis/synth.py (Supp. S2-S4, S6)
    submit_all_12d.sh         WaveDINO retrained with <= 12-day noise scenes (Supp. S1): train -> resume -> evaluate
    submit_all_cross_site.sh  each site's WaveDINO evaluated on the other site (Supp. S8)

Jobs are written to jobs/<group>/<dataset>/ and log to out/%j_%x.out (create out/ first). Every job uses the newest
run of its model type (--model-name last); pass a run folder instead to pin one. Edit SLURM_HEADER / ENV_SETUP for
your cluster.

    python util/generate_jobs.py && mkdir -p out && ./submit_all.sh
"""
import os

SLURM_HEADER = """#!/bin/bash

#SBATCH --output={output}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00         # Hours:Mins:Secs
"""

# DINOv3 is a gated model on the Hugging Face Hub: accept its licence and log in once (huggingface-cli login)
ENV_SETUP = """
echo "Job started on $(hostname), job ID $SLURM_JOB_ID, $(date +%Y_%m_%d_%H:%M:%S)"

export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_PROGRESS_BARS=1

conda activate wavedino
"""

datasets = ['laguna_del_male', 'campi_flegrei']
image_size = 128
epochs = 1000
num_workers = 5
log_frequency = 10
save_frequency = 50
result_dir = 'results'

# model type -> batch size used for the paper (one GPU per job)
models = {
    'classicunet': 20,
    'diffusion': 10,
    'swinir': 25,
    'restormer': 10,
    'wavemamba': 35,
    'wavedino-4layers': 35,          # WaveDINO, the paper model
    'wavedino-4layers-water': 35,    # ablation V1: no water mask / distance input
    'wavedino-4layers-dem': 35,      # ablation V2: no DEM conditioning
    'wavedino-4layers-ramp': 35,     # ablation V3: no phase ramp head
    'wavedino-4layers-dino': 35,     # ablation V4: no DINOv3 features
}

# analysis/synth.py analyses per model type ('unwrap' includes the clean `atm` test)
synth_analyses = {
    'wavedino-4layers': ['unwrap', 'interp', 'pure --test-limit 2000', 'grid --test-limit 4000'],
    'wavedino-4layers-ramp': ['pure --test-limit 2000'],
    'classicunet': ['unwrap'],
    'diffusion': ['unwrap'],
    'swinir': ['unwrap'],
    'restormer': ['unwrap'],
    'wavemamba': ['unwrap'],
}

MAIN = ('python main.py --dataset {dataset} --model-type {model_type} --model-name "{model_name}" '
        f'--epochs {epochs} --batch-size {{batch_size}} --image-size {image_size} --num-workers {num_workers} '
        f'--log-frequency {log_frequency} --save-frequency {save_frequency} --result-dir {result_dir} '
        '--interpolation new{extra}')
SYNTH = ('python -m analysis.synth --analysis {analysis} --dataset {dataset} --model-type {model_type} '
         f'--model-name "{{model_name}}" --batch-size {{batch_size}} --image-size {image_size} '
         f'--num-workers {num_workers} --result-dir {result_dir} --interpolation new{{extra}}')


def write_job(group, dataset, filename, commands):
    folder = os.path.join('jobs', group, dataset)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write(SLURM_HEADER.format(output='out/%j_%x.out') + ENV_SETUP + '\n' + '\n'.join(commands) + '\n')
    return path


def write_submit(filename, lines):
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n\n' + ''.join(lines))
    os.chmod(filename, 0o755)


def sbatch(job_name, path, singleton=False):
    dependency = ' --dependency=singleton' if singleton else ''
    return f'sbatch --job-name="{job_name}_$(date +%Y_%m_%d)"{dependency} {path}\nsleep 1\n'


def main_cmd(dataset, model_type, model_name='', extra=''):
    return MAIN.format(dataset=dataset, model_type=model_type, model_name=model_name, extra=extra,
                       batch_size=models[model_type.replace('-12d', '')])


# training, resuming and evaluation: one job per model and dataset; resume and evaluation jobs share the training
# job's name, so with --dependency=singleton they run after it
for group, submit, model_name, extras, singleton in [
    ('train', 'submit_all.sh', '', [''], False),
    ('train_checkpoint', 'submit_all_checkpoint.sh', 'last', [' --load-checkpoint'], True),
    ('test', 'submit_all_test.sh', 'last', [' --load-checkpoint --evaluate --synthetic-test',
                                            ' --load-checkpoint --evaluate'], True),
]:
    lines = []
    for dataset in datasets:
        for model_type in models:
            path = write_job(group, dataset, f'job_{model_type}_{dataset}.sh',
                             [main_cmd(dataset, model_type, model_name, extra) for extra in extras])
            lines.append(sbatch(f'{model_type}_{dataset}', path, singleton))
        lines.append('\n')
    write_submit(submit, lines)

# synthetic analyses
lines = []
for dataset in datasets:
    for model_type, analyses in synth_analyses.items():
        commands = []
        for analysis in analyses:
            analysis, *extra = analysis.split(' ', 1)
            commands.append(SYNTH.format(analysis=analysis, dataset=dataset, model_type=model_type, model_name='last',
                                         batch_size=models[model_type], extra=f' {extra[0]}' if extra else ''))
        path = write_job('synth_analysis', dataset, f'job_{model_type}_{dataset}.sh', commands)
        lines.append(sbatch(f'synth_{model_type}_{dataset}', path))
    lines.append('\n')
write_submit('submit_all_synth.sh', lines)

# stricter temporal-baseline cutoff: '-12d' treats 18-day pairs like long ones (only their edges are used), so the
# whole-scene noise samples are 6- and 12-day pairs only (util.datasets.get_day_interval)
lines = []
for dataset in datasets:
    model_type = 'wavedino-4layers-12d'
    for group, model_name, extra in [('train', '', ''), ('train_checkpoint', 'last', ' --load-checkpoint'),
                                     ('test', 'last', ' --load-checkpoint --evaluate --no-patches')]:
        path = write_job(group, dataset, f'job_{model_type}_{dataset}.sh',
                         [main_cmd(dataset, model_type, model_name, extra)])
        lines.append(sbatch(f'{model_type}_{dataset}', path, singleton=True))
    lines.append('\n')
write_submit('submit_all_12d.sh', lines)

# cross-site: the newest WaveDINO run of the source site, evaluated on the target site (GNSS + clean synthetic test);
# outputs are named after the source run
lines = []
for target in datasets:
    for source in datasets:
        if source == target:
            continue
        model_name = f'$(ls -d {result_dir}/{source}_{image_size}/WaveDino-4layers_network_* | tail -n 1)'
        commands = [main_cmd(target, 'wavedino-4layers', model_name, ' --load-checkpoint --evaluate --no-patches'),
                    SYNTH.format(analysis='atm', dataset=target, model_type='wavedino-4layers', model_name=model_name,
                                 batch_size=models['wavedino-4layers'], extra='')]
        path = write_job('cross_site', target, f'job_wavedino-4layers_from_{source}_on_{target}.sh', commands)
        lines.append(sbatch(f'cross_{source}_on_{target}', path))
write_submit('submit_all_cross_site.sh', lines)

print('Wrote jobs/ and submit_all*.sh')
