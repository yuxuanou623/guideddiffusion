from configs import brats_configs


def file_from_dataset(dataset_name):
    if dataset_name == "brats":
        return brats_configs.get_default_configs()
    elif dataset_name == "ldfdct":
        return brats_configs.get_default_configs()
    elif dataset_name == "oxaaa":
        return brats_configs.get_default_configs()
    else:
        raise Exception("Dataset not defined.")