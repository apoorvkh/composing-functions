import itertools
import multiprocessing as mp
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

import polars as pl
import tango.cli
import tango.common.exceptions
import tyro
from tango import Step, StepGraph, StepState
from tango.cli import tango_cli
from tqdm import tqdm

from .__tango__ import tango_executor, tango_settings, tango_workspace

__all__ = ["Experiment", "Sweep"]


@dataclass
class Experiment(ABC):
    @property
    @abstractmethod
    def step_dict(self) -> dict[str, Step]:
        raise NotImplementedError

    @property
    def dependencies(self) -> Sequence["Experiment"]:
        return []

    def results(self) -> Any:
        return {}

    def print_results(self) -> None:
        print(self.results())

    ###

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @property
    def step_graph(self) -> StepGraph:
        return StepGraph(self.step_dict)

    def step_result(self, step_name: str) -> Any:
        if not self.is_cached():
            raise ValueError(f"Step {step_name} is not cached in workspace")
        return self.step_dict[step_name].result(workspace=tango_workspace)

    def check_dependencies(self):
        dependent_experiments = self.dependencies
        not_cached = sum([not e.is_cached() for e in dependent_experiments])
        if not_cached > 0:
            print(f"{not_cached} / {len(dependent_experiments)} dependent experiments need to be run first")
            sys.exit(1)

    def _execute_step_graph(self) -> None:
        # avoid "RuntimeError: context has already been set"
        # if CLI was already initialized
        mp.set_start_method(None, force=True)

        try:
            with tango_cli(tango_settings):
                tango.cli.execute_step_graph(
                    step_graph=self.step_graph,
                    workspace=tango_workspace,
                    executor=tango_executor,
                )
        except tango.common.exceptions.CliRunError:
            pass

    def is_cached(self) -> bool:
        for s in self.step_dict.values():
            if s.CACHEABLE and s not in tango_workspace.step_cache:
                return False
        return True

    def is_running(self) -> bool:
        return any([tango_workspace.step_info(s).state == StepState.RUNNING for s in self.step_dict.values()])

    def launch(self) -> None:
        self.check_dependencies()
        print(f"\nRunning experiment: {self}\n")
        self._execute_step_graph()
        if self.is_cached():
            print("\nResults:")
            self.print_results()

    @classmethod
    @tyro.conf.configure(tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes)
    def launch_cli(cls, experiment: Self) -> None:
        return experiment.launch()

    @classmethod
    def cli(cls) -> None:
        tyro.cli(cls.launch_cli)


ExperimentT = TypeVar("ExperimentT", bound=Experiment)


class Sweep(Generic[ExperimentT], ABC):
    @property
    @abstractmethod
    def experiments(self) -> Sequence[ExperimentT]:
        raise NotImplementedError

    def results(self) -> pl.DataFrame:
        return pl.DataFrame([{**e.to_dict(), **e.results()} for e in self.experiments if e.is_cached()], strict=False)

    def print_results(self) -> None:
        print(self.results())

    ###

    @staticmethod
    def _args_product(*args):
        return itertools.product(*args)

    @staticmethod
    def _kwargs_product(**kwargs):
        # from https://stackoverflow.com/a/5228294
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))

    @property
    def num_cached(self) -> int:
        return sum([e.is_cached() for e in self.experiments])

    def cached_experiments(self) -> pl.DataFrame:
        return pl.DataFrame([{**e.to_dict(), "cached": e.is_cached()} for e in self.experiments])

    def print_incomplete(self) -> None:
        print("\nThe following experiments are incomplete and are not currently running:\n")
        for e in self.experiments:
            if not e.is_cached() and not e.is_running():
                print(e)

    def sweep(self) -> None:
        for e in tqdm(self.experiments, desc="Experiments"):
            if not e.is_cached() and not e.is_running():
                e.launch()

    @classmethod
    @tyro.conf.configure(
        tyro.conf.OmitArgPrefixes,
        tyro.conf.OmitSubcommandPrefixes,
        tyro.conf.SuppressFixed,
    )
    def run(
        cls,
        experiment_sweep: Self,
        cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run",
    ) -> None:
        pl.Config(tbl_cols=20, tbl_rows=100).__enter__()

        match cmd:
            case "run":
                experiment_sweep.sweep()
            case "count":
                print(f"# cached experiments: {experiment_sweep.num_cached} / {len(experiment_sweep.experiments)}")
            case "print-incomplete":
                experiment_sweep.print_incomplete()
            case "print-results":
                experiment_sweep.print_results()

    @classmethod
    def cli(cls) -> None:
        tyro.cli(cls.run)
