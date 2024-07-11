import json
import os
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

import metadata
import monash
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import gluonts
import wandb
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch import DeepAREstimator

# from gluonts.mx import DeepAREstimator

warnings.filterwarnings("ignore")


def makedirs_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_np_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, type):
        return obj.__name__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Handler(ABC):

    @abstractmethod
    def handle(self, data: Any) -> None:
        pass


class JsonFileHandler(Handler):
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename + ".json"

    def handle(self, data: Dict[str, Any]) -> None:
        makedirs_if_not_exists(self.path)
        with open(os.path.join(self.path, self.filename), "w") as json_file:
            json.dump(data, json_file, indent=4, default=convert_np_float)


def backtest(predictor, dataset):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print(json.dumps(agg_metrics, indent=4))

    forecast_array = make_forecast_np_array(forecasts)

    return agg_metrics, forecast_array


def make_forecast_np_array_olt(forecasts):
    return (
        torch.tensor([f.samples for f in forecasts])
        .median(dim=1)
        .values.unsqueeze(-1)
    )


def make_forecast_np_array(forecasts):
    return torch.tensor([f.quantile(0.5) for f in forecasts]).unsqueeze(-1)


class CSVFileAggregator:
    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename

    def __call__(self) -> pd.DataFrame:
        data_list = []
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                with open(os.path.join(self.path, filename), "r") as file:
                    data = json.load(file)
                    data_list.append(data)
        # Concatenate DataFrames if needed
        results = pd.DataFrame(data_list)
        results.to_csv(f"{self.path}/{self.filename}.csv", index=False)
        return results


keys = [
    "bitcoin",
    "car_parts",
    "cif_2016",
    "covid_deaths",
    "dominick",
    "electricity_daily",
    "electricity_weekly",
    "fred_md",
    "hospital",
    "kaggle_web_traffic",
    "kdd_cup",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_quarterly",
    "m3_yearly",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_weekly",
    "m4_yearly",
    "nn5_daily",
    "nn5_weekly",
    "pedestrian_counts",
    "rideshare",
    "saugeen_river_flow",
    "solar_10_minutes",
    "solar_weekly",
    "sunspot",
    "temperature_rain",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "traffic_hourly",
    "traffic_weekly",
    "us_births",
    "vehicle_trips",
    "weather",
    "australian_electricity_demand",
]

import metrics

if __name__ == "__main__":
    PATH = "results"
    data_path = (
        "/Users/garethdavies/Development/workspaces/nnts/projects/deepar/data"
    )
    # dataset_name = "traffic"
    for dataset_name in keys:
        model_name = "deepar"
        md = metadata.load(
            dataset_name,
            path=os.path.join(data_path, f"base-lstm-monash.json"),
        )
        file_path = os.path.join(data_path, md.filename)
        for seed in [42, 43, 44, 45, 46]:
            wandb.init(
                project=f"monash-gluonts-mx-{model_name}",
                name=f"{dataset_name}-{seed}",
                config=md.__dict__,
            )
            handler = JsonFileHandler(
                path=PATH, filename=f"{dataset_name}-{seed}"
            )

            train_ds, test_ds = monash.get_deep_nn_forecasts(
                dataset_name,
                md.context_length,
                file_path,
                "feed_forward",
                md.prediction_length,
                True,
            )

            torch.manual_seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed=seed, workers=True)

            lags_seq = gluonts.time_feature.lag.get_lags_for_frequency(md.freq)

            estimator = DeepAREstimator(
                context_length=md.context_length,
                prediction_length=md.prediction_length,
                freq=md.freq,
                lags_seq=lags_seq,
                trainer_kwargs={"accelerator": "cpu", "max_epochs": 100},
            )
            predictor = estimator.train(train_ds)

            gluon_metrics, y_hat = backtest(predictor, test_ds)
            y = np.stack(
                [item["target"][-md.prediction_length :] for item in test_ds]
            )

            past_data = [
                torch.tensor(item["target"][: -md.prediction_length])
                for item in test_ds
            ]

            seasonal_error = metrics.calculate_seasonal_error(
                past_data, md.seasonality
            )

            monash_metrics = metrics.calc_metrics(
                y_hat,
                torch.tensor(y).unsqueeze(-1),
                seasonal_error,
            )

            handler.handle(monash_metrics)
            wandb.log(monash_metrics)
            print(f"Done {seed}")
            wandb.finish()
        csv_aggregator = CSVFileAggregator(PATH, dataset_name)
        results = csv_aggregator()
        univariate_results = results.loc[:, ["sMAPE", "MASE"]]
        print(
            univariate_results.mean(),
            univariate_results.std(),
            univariate_results.count(),
        )
