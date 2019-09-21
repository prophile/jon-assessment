from __future__ import annotations

import abc
import argparse
import statistics
import sys
from dataclasses import dataclass, field
from math import log
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NewType,
    Optional,
    Sequence,
    Type,
)

from ruamel import yaml

SourceID = NewType("SourceID", str)
ScoringMethodName = NewType("ScoringMethodName", str)
RouteName = NewType("RouteName", str)

SCORING_METHODS_BY_NAME: Dict[ScoringMethodName, Type[ScoringMethod]] = {}


class ScoringMethod(abc.ABC):
    short_name: ScoringMethodName

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        raise NotImplementedError

    def format(self, score: float) -> str:
        return str(score)

    @classmethod
    def __init_subclass__(cls, short_name: Optional[str] = None) -> None:
        super().__init_subclass__()
        if short_name is not None:
            cls.short_name = ScoringMethodName(short_name)
            SCORING_METHODS_BY_NAME[ScoringMethodName(short_name)] = cls


class BrierScoring(ScoringMethod, short_name="brier"):
    description = "Brier score as advocated by Tetlock et al"
    format = "{:.4f}".format

    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        return (correct_prediction - 1) ** 2 + sum(x ** 2 for x in other_predictions)


class NormalisedBrierScoring(BrierScoring, short_name="brier-normalised"):
    description = "Brier score as advocated by Tetlock et al, normalised to 0-1"

    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        return 0.5 * super().score(correct_prediction, other_predictions)


class LogLossScoring(ScoringMethod, short_name="log-loss"):
    description = "Cross-entropy loss, analogous to logistic regression"
    format = "{:.3f} bits".format

    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        return -log(correct_prediction, 2) - sum(
            log(1 - x, 2) for x in other_predictions
        )


class AccuracyScoring(ScoringMethod, short_name="accuracy"):
    description = "Raw predictive accuracy"

    def format(self, score: float) -> str:
        return f"{score * 100:.1f}%"

    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        if len(other_predictions) == 0:
            # Undefined
            return correct_prediction > 0.5
        return 1.0 if correct_prediction > max(other_predictions) else 0.0


class IntuitiveScoring(ScoringMethod, short_name="intuitive"):
    description = "Intuitive (non-scientific) loss as perceived; with >=.6 being guaranteed and <.4 the opposite"

    def format(self, score: float) -> str:
        return f"{score * 100:.1f}%"

    def score(
        self, correct_prediction: float, other_predictions: Sequence[float]
    ) -> float:
        return 1.0 if correct_prediction >= 0.6 else 0.0


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
        type=ScoringMethodName,
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
        new_id = SourceID(f"{non_committal}_non_committal")
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
                other_predictions: List[float] = []
                for (
                    predicted_outcome_name,
                    prediction_prob,
                ) in predicted_outcomes.items():
                    if predicted_outcome_name == prediction["outcome"]:
                        continue
                    other_predictions.append(prediction_prob)
                prediction_prob = predicted_outcomes[prediction["outcome"]]
                source.scores.append(scorer.score(prediction_prob, other_predictions))

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
