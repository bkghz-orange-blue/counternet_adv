from counterfactual.adversarial_experiment import _calculate_validity_matrix, ExperimentLoggerWanb
from counterfactual.evaluate import load_trained_model, model_cf_gen
from counterfactual.net import AdvCounterfactualModel

if __name__ == '__main__':
    config = {
        "data_dir": "assets/data/adv/loan/org.csv",
        "data_name": "loan",
        "lr": 0.003,
        "batch_size": 128,
        "lambda_1": 1.0,
        "lambda_2": 0.01,
        "lambda_3": 0.2,
        "threshold": 1.0,
        "continous_cols": [
            "NoEmp", "NewExist", "CreateJob", "RetainedJob", "DisbursementGross", "GrAppv", "SBA_Appv"
        ],
        "discret_cols": [
            "State", "Term", "UrbanRural", "LowDoc", "Sector_Points"
        ],
        "enc_dims": [
            110,
            200,
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

    model_path_list = [
        "log/adv/loan/year=1994/version_0/checkpoints/epoch=58-step=5663.ckpt",
        "log/adv/loan/year=1995/version_0/checkpoints/epoch=88-step=12192.ckpt",
        "log/adv/loan/year=1996/version_0/checkpoints/epoch=137-step=16835.ckpt",
        "log/adv/loan/year=1997/version_0/checkpoints/epoch=124-step=14624.ckpt",
        "log/adv/loan/year=1998/version_0/checkpoints/epoch=149-step=17249.ckpt",
        "log/adv/loan/year=1999/version_0/checkpoints/epoch=138-step=16818.ckpt",
        "log/adv/loan/year=2000/version_0/checkpoints/epoch=132-step=16225.ckpt",
        "log/adv/loan/year=2001/version_0/checkpoints/epoch=135-step=16727.ckpt",
        "log/adv/loan/year=2002/version_0/checkpoints/epoch=148-step=21753.ckpt",
        "log/adv/loan/year=2003/version_0/checkpoints/epoch=65-step=12869.ckpt",
        "log/adv/loan/year=2004/version_0/checkpoints/epoch=121-step=28913.ckpt",
        "log/adv/loan/year=2005/version_0/checkpoints/epoch=148-step=42315.ckpt",
        "log/adv/loan/year=2006/version_0/checkpoints/epoch=143-step=43199.ckpt",
        "log/adv/loan/year=2007/version_0/checkpoints/epoch=149-step=44849.ckpt",
        "log/adv/loan/year=2008/version_0/checkpoints/epoch=144-step=23344.ckpt",
        "log/adv/loan/year=2009/version_0/checkpoints/epoch=141-step=9229.ckpt"
    ]

    cf_results = {}
    model_list = []

    for i, m_path in enumerate(model_path_list):
        model = load_trained_model(module=AdvCounterfactualModel, checkpoint_path=m_path)
        cf_results[f'm_{i}'] = model_cf_gen(model) #_useful_result(model_cf_gen(m))
        model_list.append(model)

    # calculate the validity matrix
    print("calculating validity matrix...")
    # print(cf_results)
    validity_matrix = _calculate_validity_matrix(model_list, cf_results)

    # store result
    results = {
        'adv': True,
        'data_name': "loan",
        'hparams': config,
        'cf_results': cf_results,
        'validity_matrix': validity_matrix,
        'model_list': model_path_list,
        # 'result_path': path
    }
    dir_path = ExperimentLoggerWanb("test").store_results(results)
    print(f"Results stored at {dir_path}.")
