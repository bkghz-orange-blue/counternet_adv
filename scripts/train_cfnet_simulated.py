from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import train_model
from counterfactual.evaluate import load_trained_model, model_cf_gen
from counterfactual.net import AdvCounterfactualModel, BaselineModel, CounterfactualModel
from counterfactual.training_module import BaseModule, CounterfactualTrainingModule
from sklearn.linear_model import LogisticRegression
from counterfactual.adversarial_experiment import experiment_step, avg_decrease

torch.autograd.set_detect_anomaly(True)


def generate_data(data_path: Path, means, stds):
    x_1_data = {
        f'x{i+1}': np.random.normal(-mean, std, 1000) for i, (mean, std) in enumerate(zip(means, stds))
    }
    x_1_data.update({'y': np.ones((1000,))})
    x_0_data = {
        f'x{i+1}': np.random.normal(mean, std, 1000) for i, (mean, std) in enumerate(zip(means, stds))
    }
    x_0_data.update({'y': np.zeros((1000,))})
    X = pd.concat((
        pd.DataFrame(x_1_data), pd.DataFrame(x_0_data)
    ))
    X = shuffle(X)
    X.to_csv(data_path, index=False)
    return X


def generate_shift_data(dimensions: int, data_dir: Path):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        data_dir.mkdir()
    org_data = generate_data(data_dir / Path('original.csv'), 
        means=[2 for _ in range(dimensions)], stds=[0.5 for _ in range(dimensions)])
    upt_data = generate_data(data_dir / Path('updated.csv'), 
        means=[1. + (random.random() / 5) for _ in range(dimensions)], stds=[1. for _ in range(dimensions)])
    org_model = LogisticRegression().fit(org_data[[f'x{i+1}' for i in range(dimensions)]], org_data['y'])
    upt_model = LogisticRegression().fit(upt_data[[f'x{i+1}' for i in range(dimensions)]], upt_data['y'])
    print(f"org_model: {org_model.coef_}")
    print(f"upt_model: {upt_model.coef_}")
    
    
t_config = {
    "max_epochs": 1,
    "gpus": 1,
    "deterministic": True,
    "log_every_n_steps": 1,
    "automatic_optimization": False
}


def simulated_experiment(n_features: int):
    generate_shift_data(n_features, data_dir='assets/data/adv/simulated')
    config = {
        "data_dir": "assets/data/adv/simulated/original.csv",
        "data_name": "simulated",
        "lr": 0.003,
        "batch_size": 128,
        "lambda_1": 1.0,
        "lambda_2": 0.01,
        "lambda_3": 0.2,
        "threshold": 1.0,
        "continous_cols": [
            f'x{i + 1}' for i in range(n_features)
        ],
        "discret_cols": [],
        "enc_dims": [
            n_features,
            100,
            10
        ],
        "dec_dims": [
            10,
            10
        ],
        "exp_dims": [
            10,
            10
        ],
        "loss_func_1": "mse",
        "loss_func_2": "mse",
        "loss_func_3": "mse",
    }
    data_dir_list = [
        "assets/data/adv/simulated/original.csv",
        "assets/data/adv/simulated/updated.csv"
    ]
    no_adv_result = experiment_step(config, t_config, data_dir_list, CounterfactualModel)
    adv_result = experiment_step(config, t_config, data_dir_list, AdvCounterfactualModel)
    return no_adv_result, adv_result

def matrix_list(results):
    return [result['validity_matrix'] for result in results]


def avg_validity_matrix(results):
    matrix_all = matrix_list(results)
    return np.average(matrix_all, axis=0), np.std(matrix_all, axis=0)

def avg_cf_result(results):
    # TODO
    pass

def main():
    no_adv_results = []
    adv_results = []

    for n_features in [2, 5, 10, 20, 30, 40, 50]:
        no_adv_result, adv_result = simulated_experiment(n_features)
        no_adv_results.append(no_adv_result)
        adv_results.append(adv_result)

    print(f"[no adv] validity matrix: \n"
        f"{avg_validity_matrix(no_adv_results)}")
    print(f"[bi-adv] validity matrix: \n"
        f"{avg_validity_matrix(adv_results)}")

    print(f"[no adv] validity matrix: \n"
        f"{matrix_list(no_adv_results)}")
    print(f"[bi-adv] validity matrix: \n"
        f"{matrix_list(adv_results)}")

    

if __name__ == "__main__":
    main()
