"""Compositional task definitions and utilities."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

NodeT = Literal["x", "Fx", "Gx", "GFx", "FGx"]
HopT = tuple[NodeT, NodeT]


@dataclass(frozen=True)
class CompositionalTask:
    x: str
    Fx: str
    Gx: str | None
    GFx: str
    FGx: str | None

    def get(self, node_type: NodeT, leading_space: bool = False) -> str:
        """Get the string value for a specific node type."""
        attr = getattr(self, node_type)
        assert attr is not None
        if leading_space:
            return f" {attr}"
        return attr

    @staticmethod
    def overlap(a: "CompositionalTask", b: "CompositionalTask") -> bool:
        """Check if two tasks share any values across their nodes."""
        return len(({a.x, a.Fx, a.Gx, a.GFx, a.FGx} & {b.x, b.Fx, b.Gx, b.GFx, b.FGx}) - {None}) > 0


##

TaskT = Literal[
    # antonym translation
    "antonym-spanish",
    "antonym-german",
    "antonym-french",
    # wikidata relations
    "book-author-birthyear",
    "song-artist-birthyear",
    "landmark-country-capital",
    "park-country-capital",
    "movie-director-birthyear",
    "person-university-year",
    "person-university-founder",
    "product-company-ceo",
    "product-company-hq",
    # arithmetic
    "plus-ten-times-two",
    "plus-hundred-times-two",
    "mod-twenty-times-two",
    "word-int-times-two",
    "word-substring-reverse",
    "rgb-rot120-name",
]


@dataclass
class Task:
    """Wrapper for compositional task datasets and properties."""

    task_name: TaskT

    def build_dataset(self) -> list[CompositionalTask]:
        """Build dataset of compositional tasks for this task."""
        from . import arithmetic, translation, wikidata

        match self.task_name:
            case "antonym-spanish":
                dataset = translation.antonym_translation(language="es")
            case "antonym-german":
                dataset = translation.antonym_translation(language="de")
            case "antonym-french":
                dataset = translation.antonym_translation(language="fr")
            case "book-author-birthyear":
                dataset = wikidata.book_author_birthyear()
            case "song-artist-birthyear":
                dataset = wikidata.song_artist_birthyear()
            case "landmark-country-capital":
                dataset = wikidata.landmark_country_capital()
            case "park-country-capital":
                dataset = wikidata.park_country_capital()
            case "movie-director-birthyear":
                dataset = wikidata.movie_director_birthyear()
            case "person-university-year":
                dataset = wikidata.person_university_year()
            case "person-university-founder":
                dataset = wikidata.person_university_founder()
            case "product-company-ceo":
                dataset = wikidata.product_company_ceo()
            case "product-company-hq":
                dataset = wikidata.product_company_hq()
            case "plus-ten-times-two":
                dataset = arithmetic.plus_ten_times_two()
            case "plus-hundred-times-two":
                dataset = arithmetic.plus_hundred_times_two()
            case "mod-twenty-times-two":
                dataset = arithmetic.mod_twenty_times_two()
            case "word-int-times-two":
                dataset = arithmetic.word_int_times_two()
            case "word-substring-reverse":
                dataset = arithmetic.word_substr_reverse()
            case "rgb-rot120-name":
                dataset = arithmetic.rgb_rotation_name(degrees=120)

        # remove duplicates (over all fields)
        dataset = list(set(dataset))
        # remove entries with x that appears more than once
        occurences = defaultdict(int)
        for task in dataset:
            occurences[task.x] += 1
        dataset = [task for task in dataset if occurences[task.x] == 1]
        return dataset

    @property
    def nodes(self) -> list[NodeT]:
        """Get the node types involved in this task."""
        match self.task_name:
            # one-way
            case (
                "book-author-birthyear"
                | "song-artist-birthyear"
                | "landmark-country-capital"
                | "park-country-capital"
                | "movie-director-birthyear"
                | "person-university-year"
                | "person-university-founder"
                | "product-company-ceo"
                | "product-company-hq"
            ):
                return ["x", "Fx", "GFx"]
            # commutative
            case "antonym-spanish" | "antonym-german" | "antonym-french" | "word-int-times-two" | "rgb-rot120-name":
                return ["x", "Fx", "Gx", "GFx"]
            # non-commutative with correct hop ("Gx", "GFx")
            case "plus-ten-times-two" | "plus-hundred-times-two" | "mod-twenty-times-two" | "word-substring-reverse":
                return ["x", "Fx", "Gx", "GFx", "FGx"]

    @property
    def correct_intermediate_nodes(self) -> list[NodeT]:
        match self.task_name:
            # one-way
            case (
                "book-author-birthyear"
                | "song-artist-birthyear"
                | "landmark-country-capital"
                | "park-country-capital"
                | "movie-director-birthyear"
                | "person-university-year"
                | "person-university-founder"
                | "product-company-ceo"
                | "product-company-hq"
            ):
                return ["Fx"]
            # commutative
            case "antonym-spanish" | "antonym-german" | "antonym-french" | "word-int-times-two" | "rgb-rot120-name":
                return ["Fx", "Gx"]
            # non-commutative with correct hop ("Gx", "GFx")
            case "plus-ten-times-two" | "plus-hundred-times-two" | "mod-twenty-times-two" | "word-substring-reverse":
                return ["Fx", "Gx"]

    @property
    def correct_hops(self) -> list[HopT]:
        """Get the hops that are part of the correct computation for this task."""
        match self.task_name:
            # one-way
            case (
                "book-author-birthyear"
                | "song-artist-birthyear"
                | "landmark-country-capital"
                | "park-country-capital"
                | "movie-director-birthyear"
                | "person-university-year"
                | "person-university-founder"
                | "product-company-ceo"
                | "product-company-hq"
            ):
                return [("x", "Fx"), ("Fx", "GFx")]
            # commutative
            case "antonym-spanish" | "antonym-german" | "antonym-french" | "word-int-times-two" | "rgb-rot120-name":
                return [("x", "Fx"), ("Fx", "GFx"), ("x", "Gx"), ("Gx", "GFx")]
            # non-commutative with correct hop ("Gx", "GFx")
            case "plus-ten-times-two" | "plus-hundred-times-two" | "mod-twenty-times-two" | "word-substring-reverse":
                return [("x", "Fx"), ("Fx", "GFx"), ("x", "Gx"), ("Gx", "GFx")]

    @property
    def incorrect_hops(self) -> list[HopT]:
        """Get the hops that do not lead to the correct computation of this task."""
        match self.task_name:
            # one-way
            case (
                "book-author-birthyear"
                | "song-artist-birthyear"
                | "landmark-country-capital"
                | "park-country-capital"
                | "movie-director-birthyear"
                | "person-university-year"
                | "person-university-founder"
                | "product-company-ceo"
                | "product-company-hq"
            ):
                return []
            # commutative
            case "antonym-spanish" | "antonym-german" | "antonym-french" | "word-int-times-two" | "rgb-rot120-name":
                return []
            # non-commutative with correct hop ("Gx", "GFx")
            case "plus-ten-times-two" | "plus-hundred-times-two" | "mod-twenty-times-two" | "word-substring-reverse":
                return [("Gx", "FGx")]

    def leading_space(self, node_type: NodeT) -> bool:
        """
        Determine if this node type should have a leading space in prompts.
        Because of tokenization, basically: which fields (per task) correspond to numbers?
        """
        match self.task_name, node_type:
            case (
                ("book-author-birthyear", "GFx")
                | ("song-artist-birthyear", "GFx")
                | ("movie-director-birthyear", "GFx")
                | ("person-university-year", "GFx")
                | ("plus-ten-times-two", _)
                | ("plus-hundred-times-two", _)
                | ("mod-twenty-times-two", _)
                | ("word-int-times-two", "Fx" | "GFx" | "FGx")
            ):
                return False
            case (
                ("antonym-spanish", _)
                | ("antonym-german", _)
                | ("antonym-french", _)
                | ("book-author-birthyear", _)
                | ("song-artist-birthyear", _)
                | ("landmark-country-capital", _)
                | ("park-country-capital", _)
                | ("movie-director-birthyear", _)
                | ("person-university-year", _)
                | ("person-university-founder", _)
                | ("product-company-ceo", _)
                | ("product-company-hq", _)
                | ("word-int-times-two", _)
                | ("word-substring-reverse", _)
                | ("rgb-rot120-name", _)
            ):
                return True

    def trailing_space_in_query(self, pred_type: NodeT) -> bool:
        """Query should have trailing space if prediction string does not have leading space."""
        return not self.leading_space(pred_type)


__all__ = ["NodeT", "HopT", "CompositionalTask", "TaskT", "Task"]
