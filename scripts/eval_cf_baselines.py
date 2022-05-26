from counterfactual.import_essentials import *
from counterfactual.train import train_model
from counterfactual.evaluate import load_trained_model, model_cf_gen
from counterfactual.net import AdvCounterfactualModel, BaselineModel, CounterfactualModel
from counterfactual.training_module import BaseModule, CounterfactualTrainingModule
from counterfactual.adversarial_experiment import experiment_step, local_explainer_experiment_step, ExperimentLoggerWanb
from counterfactual.adversarial_baselines import ROAR#, VanillaCF
from counterfactual.utils import load_json
from .utils import get_data, DATASET_NAMES
from counterfactual.baseline_cfs import VanillaCF
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import argparse


_MODEL_NAMES = ['VanillaCF', 'ROAR']
_MODEL_DICT = {
    "VanillaCF": {
        'module': VanillaCF,
        'adv': False
    },
    "ROAR": {
        'module': ROAR,
        'adv': True
    }
}

t_config = {
    "max_epochs": 20,
    "gpus": 0,
    "deterministic": True,
    "benchmark": True,
    # "automatic_optimization": False
}


def main(args):
    m_config, data_dir_list = get_data(args.data_name)
    t_config = load_json("assets/configs/adv/t_config.json")
    t_config['automatic_optimization'] = True
    # local_explainer_experiment_step(
    #     default_model_config = config, 
    #     t_config = t_config, 
    #     data_dir_list = data_dir_list,
    #     pred_module = BaselineModel,
    #     cf_params = {'max_delta': 0.1, 'n_iters': 50},
    #     cf_module = ROAR,
    #     is_parallel = False,
    #     test_size = None,           # use all test dataset
    #     use_prev_model_weights = False,
    #     return_best_model = False,  # return last model by default
    #     experiment_logger = ExperimentLoggerWanb(logger_name="roar")
    # )

    m_config['adv'] = _MODEL_DICT[args.model_name]['adv']
    local_explainer_experiment_step(
        default_model_config = m_config, 
        t_config = t_config, 
        data_dir_list = data_dir_list,
        pred_module = BaselineModel,
        cf_params = {'n_iters': args.n_iters},
        cf_module = _MODEL_DICT[args.model_name]['module'],
        is_parallel = False,
        test_size = None,
        use_prev_model_weights = False,
        return_best_model = False, # return last model by default
        experiment_logger=ExperimentLoggerWanb(
            logger_name=f"{args.model_name}-{args.data_name}"
        )
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', 
                        type=str, 
                        default='german_credit', 
                        choices=DATASET_NAMES)
    parser.add_argument('--model_name', 
                        type=str, 
                        default='ROAR',
                        choices=_MODEL_NAMES)
    parser.add_argument('--n_iters', 
                        type=int, 
                        default=50)
    args = parser.parse_args()

    main(args)
