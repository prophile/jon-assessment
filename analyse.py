from __future__ import annotations

from typing import Type, List, Iterable, Callable, Dict, Optional, NewType
from ruamel import yaml
import argparse
from dataclasses import dataclass, field
from math import log
import sys
import statistics
import abc


SourceID = NewType("SourceID", str)
RouteName = NewType("RouteName", str)


@dataclass(frozen=True)
class ScoringMethod:
    short_name: str
    description: str
    score: Callable[[float], float]
    output_format: str


SCORING_METHODS_BY_NAME: Dict[str, ScoringMethod] = {}


class ScoringMethod(abc.ABC):
    short_name: str

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def score(prediction: float) -> float:
        raise NotImplementedError

    def format(self, score: float) -> str:
        return str(score)

    @classmethod
    def __init_subclass__(cls, short_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if short_name is not None:
            cls.short_name = short_name
            SCORING_METHODS_BY_NAME[short_name] = cls


class BrierScoring(ScoringMethod, short_name="brier"):
    description = "Brier score as advocated by Tetlock et al"
    format = "{:.4f}".format

    def score(self, prediction: float) -> float:
        return (1 - prediction) ** 2


class LogLossScoring(ScoringMethod, short_name="log-loss"):
    description = "Cross-entropy loss, analogous to logistic regression"
    format = "{:.3f} bits".format

    def score(self, prediction: float) -> float:
        return -log(prediction, 2)


class AccuracyScoring(ScoringMethod, short_name="accuracy"):
    description = "Raw predictive accuracy at >.5"

    def format(self, score: float) -> str:
        return f"{score * 100:.1f}%"

    def score(self, prediction: float) -> float:
        return 1.0 if prediction > 0.5 else 0.0


class IntuitiveScoring(ScoringMethod, short_name="intuitive"):
    description = "Intuitive (non-scientific) loss as perceived; with >=.6 being guaranteed and <.4 the opposite"

    def format(self, score: float) -> str:
        return f"{score * 100:.1f}%"

    def score(self, prediction: float) -> float:
        return 1.0 if prediction >= 0.6 else 0.0


DEFAULT_SCORING_METHOD: Type[ScoringMethod] = BrierScoring


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        metavar="PREDICTIONS_FILE",
        type=argparse.FileType("r"),
        help="Predictions file to read.",
    )
    parser.add_argument(
        "--scoring",
        default=DEFAULT_SCORING_METHOD,
        choices=[x for x in SCORING_METHODS_BY_NAME.keys()],
        help="Scoring method to use.",
    )
    parser.add_argument(
        "--add-non-committal",
        dest="non_committal",
        default=[],
        action="append",
        help="Add a non-committal version of a source.",
    )
    return parser


@dataclass
class Source:
    id: SourceID
    name: str
    num_predictions: int = 0
    num_missed: int = 0
    scores: List[float] = field(default_factory=list)

    def get_prediction(
        self,
        route_name: RouteName,
        prediction_data: Dict[SourceID, Dict[RouteName, float]],
    ) -> Optional[float]:
        try:
            return prediction_data[self.id][route_name]
        except KeyError:
            return None


class NonCommittalSource(Source):
    """A modifier source which identifies the same outcomes but assigns each equal probability."""

    original_id: SourceID

    def __init__(self, *, id: SourceID, name: str, original_id: SourceID) -> None:
        super().__init__(id=id, name=name)
        self.original_id = original_id

    def get_prediction(
        self,
        route_name: RouteName,
        prediction_data: Dict[SourceID, Dict[RouteName, float]],
    ) -> Optional[float]:
        identified_routes = prediction_data[self.original_id]
        if route_name not in identified_routes:
            return None
        return 1 / len(identified_routes)


def main(args: Iterable[str] = sys.argv[1:]) -> None:
    options = argument_parser().parse_args(list(args))

    prediction_data = yaml.safe_load(options.predictions)

    scorer: ScoringMethod = SCORING_METHODS_BY_NAME[options.scoring]()

    sources = {
        key: Source(id=key, name=value["name"])
        for key, value in prediction_data["sources"].items()
    }

    for non_committal in options.non_committal:
        new_id = f"{non_committal}_non_committal"
        sources[new_id] = NonCommittalSource(
            id=new_id,
            name=f"Derived: As {sources[non_committal].name}, but with equal probability on each identified outcome",
            original_id=non_committal,
        )

    for prediction in prediction_data["predictions"]:
        print(f"Processing {prediction['name']}", file=sys.stderr)

        if prediction["outcome"] not in prediction["routes"]:
            print("  → no useful information, skipping", file=sys.stderr)
            continue

        # Find all sources who made predictions here
        raw_predictions_by_named_source: Dict[SourceID, Dict[RouteName, float]] = {}

        for route_name, route in prediction["routes"].items():
            for source_name, prediction_prob in route["predicted"].items():
                raw_predictions_by_named_source.setdefault(source_name, {})[
                    RouteName(route_name)
                ] = prediction_prob

        predictors: Dict[SourceID, Dict[RouteName, float]] = {}
        for source in sources.values():
            source_predictions: Dict[RouteName, float] = {}
            for route_name in prediction["routes"].keys():
                prediction_prob = source.get_prediction(
                    RouteName(route_name), raw_predictions_by_named_source
                )

                if prediction_prob is not None:
                    source_predictions[route_name] = prediction_prob

            if source_predictions:
                predictors[source.id] = source_predictions

        for predictor, predicted_outcomes in predictors.items():
            source = sources[predictor]
            source.num_predictions += 1
            if prediction["outcome"] not in predicted_outcomes:
                source.num_missed += 1
            else:
                prediction_prob = predicted_outcomes[prediction["outcome"]]
                source.scores.append(scorer.score(prediction_prob))

    for source in sources.values():
        print(source.name)
        print(f"   Predictions made: {source.num_predictions}")
        print(f"   Predictions unforeseen: {source.num_missed}")

        if not source.scores:
            print("   Unable to produce a meaningful score")
            continue

        score = statistics.mean(source.scores)
        score_fmt = scorer.format(score)

        print(f"   Score: {score_fmt}")


if __name__ == "__main__":
    main()
