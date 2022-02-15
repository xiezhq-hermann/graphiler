from graphiler.utils import homo_dataset, hetero_dataset, load_data


if __name__ == "__main__":
    for d in homo_dataset:
        print("Start preparing {} dataset".format(d))
        load_data(d)
        print("Dataset {} successfully downloaded.".format(d))

    for d in hetero_dataset:
        print("Start preparing {} dataset".format(d))
        load_data(d)
        print("Dataset {} successfully downloaded.".format(d))
