import pandas as pd
import dvc.api


def get_data(path,repo,version):
    data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
    )
    df = pd.read_csv(data_url)
    return df