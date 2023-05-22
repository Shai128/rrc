import argparse
import time
from itertools import product

from run_experiment import run_experiment


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1, help='')
args = parser.parse_args()

def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))
"""
srun -c 3 --exclude=dym-lab --gres=gpu:1 -J plsNoKil python run_all_depth_experiments.py --seed 0 &
srun -c 3 --exclude=dym-lab --gres=gpu:1 -J plsNoKil python run_all_depth_experiments.py --seed 1 &
"""
#


processes_to_run_in_parallel = 1

seeds = list(range(0, 10))
if args.seed == 0:
    seeds = list(range(0, 5))
elif args.seed == 1:
    seeds = list(range(5, 10))


params = {
    'main_program_name': ['multi_calibration_depth'],
    'seed': seeds,
    'dataset_name': ['KITTI'],
    'ds_type': ['REAL'],
    'max_data_size': [10000],
    'backbone': ['res101'],
    'trained_model_path': ['../saved_models/Leres'],
    'base_save_path': ['..'],
    'base_data_path': ['datasets/real_data/depths'],
    'annotations_path': ['annotations/train_annotations_onlyvideos.json'],
    'offline_train_ratio': [0.6],
    'training_epochs': [60],
    'std_method': ['baseline', 'residual_magnitude', 'previous_residual_with_alignment']
}


params = list(cartesian_product(params))

processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))
run_on_slurm = False
cpus = 2
gpus = 0
if __name__ == '__main__':

    print("jobs to do: ", len(params))
    # initializing processes_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        main_program_name = curr_params['main_program_name']
        curr_params.pop('main_program_name')
        p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                           cpus=cpus, gpus=gpus)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                main_program_name = curr_params['main_program_name']
                curr_params.pop('main_program_name')
                p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                                   cpus=cpus, gpus=gpus)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
            time.sleep(10)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")


