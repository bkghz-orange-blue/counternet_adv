from counterfactual.utils import load_json


DATASET_NAMES = ['loan', 'german_credit', 'student']
DATA_DIR_LIST_DICT = {
    'loan': [
        f"assets/data/adv/loan/year={year}.csv" for year in range(1994, 2010)
    ],
    'german_credit': [
        "assets/data/adv/german_credit/org.csv", "assets/data/adv/german_credit/upt.csv"
    ],
    "student": [
        "assets/data/adv/student/gp.csv", "assets/data/adv/student/ms.csv"
    ]
}

def get_data(data_name):
    if data_name not in DATASET_NAMES:
        raise ValueError(f"`data_name` should be one of `{DATASET_NAMES}`, but got `{data_name}`")
    # return {
    #     'm_config': load_json(f"assets/configs/adv/{data_name}.json"),
    #     "data_dir_list": DATA_DIR_LIST_DICT[data_name]
    # }
    return (
        load_json(f"assets/configs/adv/{data_name}.json"), DATA_DIR_LIST_DICT[data_name]
    ) 