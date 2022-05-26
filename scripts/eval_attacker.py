from counterfactual.import_essentials import *
from counterfactual.train import train_model
from counterfactual.evaluate import load_trained_model, model_cf_gen
from counterfactual.net import AdvCounterfactualModel, BaselineModel, CounterfactualModel
from counterfactual.training_module import BaseModule, CounterfactualTrainingModule
from counterfactual.adversarial_experiment import _train_models, _aggregate_default_data_encoders
from counterfactual.utils import l_inf_proj
from torchmetrics.functional.classification import accuracy
import higher

from counterfactual.utils import use_grad
from .utils import *
import argparse


_MODEL_NAMES = ['AdvCounterNet', 'CounterNet']
_MODEL_DICT = {
    "CounterNet": {
        'module': CounterfactualModel,
        'adv': False
    },
    "AdvCounterNet": {
        'module': AdvCounterfactualModel,
        'adv': True
    }
}


def train_model_list(m_config, t_config, data_dir_list, module, tb_logger_dir):
    m_config = deepcopy(m_config)
    default_model_config = _aggregate_default_data_encoders(m_config, data_dir_list)
    model_list = _train_models(
        default_model_config, t_config, data_dir_list, module, 
        use_prev_model_weights=False, return_best_model=False, 
        tb_logger_dir=tb_logger_dir
    )
    return model_list


def attack(model_list: List[AdvCounterfactualModel], n_steps: int, epsilon: float):
    rob_val_list = []
    for m in tqdm(model_list, desc=f"n_steps={n_steps}; epsilon={epsilon}"):
        cf_ys, y_primes = [], []
        test_loader = m.test_dataloader()
        for batch in test_loader:
            x, y = batch
            m.model_dup.load_state_dict(m.model.state_dict())
            m.n_steps = n_steps
            m.epsilon = epsilon
            m.bilevel_adv_training(batch, None)
            use_grad(m, requires_grad=False)
            y_hat, cf, cf_y, y_prime = m._perturb_input(x, torch.zeros((1)))
            cf_ys.append(cf_y)
            y_primes.append(y_prime)
        cf_ys = torch.cat(tuple(cf_ys), dim=0)
        y_primes = torch.cat(tuple(y_primes), dim=0)
        rob_val_list.append(accuracy(cf_ys, y_primes.int()))
    return rob_val_list


def main(args):
    m_config, data_dir_list = get_data(args.data_name)
    t_config = load_json("assets/configs/adv/t_config.json")
    t_config['max_epochs'] = args.epochs

    m_config['batch_size'] = args.batch_size
    m_config['adv'] = _MODEL_DICT[args.model_name]['adv']
    m_config['epsilon'] = args.epsilon
    m_config['n_steps'] = 0
    m_config['adv_loss_func'] = 'mse'
    m_config['eps_scheduler'] = args.eps_scheduler

    # model_list = train_model_list(
    #     m_config,
    #     t_config,
    #     data_dir_list,
    #     AdvCounterfactualModel,
    #     tb_logger_dir=f"CounterNet/",
    # )
    model_path_list = [
        Path(f"assets/weights/{args.model_name}_loan/{year}").glob('*.ckpt').__next__() for year in range(1994, 2010)
    ]
    model_list = [
        load_trained_model(module=AdvCounterfactualModel, checkpoint_path=str(model_path))
            for model_path in model_path_list 
    ]

    n_steps = [1, 2, 3, 5, 7, 10, 13, 15, 20]
    epsilons = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

    attacker_res = {}
    
    # for eps in epsilons:
    for step in n_steps:
        rob_val_list = attack(model_list, n_steps=step, epsilon=0.1)
        attacker_res.update({
            f"{step}": {
                'mean': np.mean(rob_val_list),
                'std': np.std(rob_val_list)
            }
        })
        # print(f"Robust Validity: {rob_val_list}")
        # print(f"mean: {np.mean(rob_val_list)}; std: {np.std(rob_val_list)}")
    pd.DataFrame.from_dict(attacker_res, orient='index')\
        .to_csv(f"assets/result/{args.data_name}-{args.model_name}-attack-l2.csv")

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
    parser.add_argument('--eps_scheduler',
                        type=str,
                        default="linear",
                        choices=["linear", "static", "log"])
    parser.add_argument('--debug',
                        type=bool,
                        default=False)
    args = parser.parse_args()

    main(args)
