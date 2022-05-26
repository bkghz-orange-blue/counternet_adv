from counterfactual.import_essentials import *
from counterfactual.train import train_model
from counterfactual.evaluate import load_trained_model, model_cf_gen
from counterfactual.net import AdvCounterfactualModel, AdvCounterfactualFramework, CounterfactualModel
from counterfactual.training_module import BaseModule, CounterfactualTrainingModule
from counterfactual.adversarial_experiment import experiment_step, avg_decrease, ExperimentLoggerWanb
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from .utils import *
import argparse


_MODEL_NAMES = ['AdvCounterNet', 'CounterNet', 'AdvCFramework']
_MODEL_DICT = {
    "CounterNet": {
        'module': CounterfactualModel,
        'adv': False
    },
    "AdvCounterNet": {
        'module': AdvCounterfactualModel,
        'adv': True
    },
    "AdvCFramework": {
        'module': AdvCounterfactualFramework,
        'adv': True
    }
}


def main(args):
    m_config, data_dir_list = get_data(args.data_name)
    t_config = load_json("assets/configs/adv/t_config.json")
    t_config['max_epochs'] = args.epochs

    # model_path_list = [
    #     Path(f"assets/weights/loan/{year}").glob('*.ckpt').__next__() for year in range(1994, 2010)
    # ]
    # m_config['adv_val_models'] = [
    #     load_trained_model(module=AdvCounterfactualModel, checkpoint_path=str(model_path))
    #         for model_path in model_path_list 
    # ]
    m_config['batch_size'] = args.batch_size
    m_config['adv'] = _MODEL_DICT[args.model_name]['adv']
    m_config['epsilon'] = args.epsilon
    m_config['n_steps'] = args.n_steps
    m_config['adv_loss_func'] = 'mse'
    m_config['eps_scheduler'] = args.eps_scheduler
    m_config['sample_frac'] = args.sample_frac
    # set logger
    experiment_logger = ExperimentLoggerWanb(
            logger_name=f"{args.model_name}-{args.data_name}-{args.eps_scheduler}"
    ) if not args.debug else None
    experiment_step(
        m_config,
        t_config,
        data_dir_list,
        _MODEL_DICT[args.model_name]['module'],
        return_best_model=False,
        tb_logger_dir=f"{args.model_name}-{args.data_name}-{args.eps_scheduler}/",
        experiment_logger=experiment_logger
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', 
                        type=str, 
                        default='loan', 
                        choices=DATASET_NAMES)
    parser.add_argument('--model_name', 
                        type=str, 
                        default='AdvCounterNet',
                        choices=_MODEL_NAMES)
    parser.add_argument('--n_steps',
                        type=int,
                        default=7)
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--sample_frac',
                        type=float,
                        nargs='?',
                        const=None,
                        default=None)
    parser.add_argument('--eps_scheduler',
                        type=str,
                        default="linear",
                        choices=["linear", "static", "log"])
    parser.add_argument('--debug',
                        type=bool,
                        default=False)
    args = parser.parse_args()

    main(args)
