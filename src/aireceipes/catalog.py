from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPES_DIR = PROJECT_ROOT / "recipes"


@dataclass(frozen=True)
class Recipe:
    id: str
    category: str
    name: str
    description: str
    tags: list[str]
    classification: dict[str, Any]
    runtime: dict[str, Any]
    parameters: dict[str, Any]
    dataset: dict[str, Any]
    kpis: dict[str, Any]
    path: Path = field(compare=False)

    @classmethod
    def from_file(cls, path: Path) -> "Recipe":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in ("id", "category", "name", "description", "runtime", "dataset") if key not in data]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Recipe {path} is missing required field(s): {joined}")

        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Recipe {path} field 'tags' must be a list of strings")

        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            name=str(data["name"]),
            description=str(data["description"]),
            tags=tags,
            classification=dict(data.get("classification", {})),
            runtime=dict(data["runtime"]),
            parameters=dict(data.get("parameters", {})),
            dataset=dict(data["dataset"]),
            kpis=dict(data.get("kpis", {})),
            path=path,
        )


class Catalog:
    def __init__(self, recipes_dir: str | Path | None = None) -> None:
        self.recipes_dir = Path(recipes_dir) if recipes_dir is not None else DEFAULT_RECIPES_DIR

    def list(self, category: str | None = None) -> list[Recipe]:
        if not self.recipes_dir.exists():
            return []
        recipes = [Recipe.from_file(path) for path in sorted(self.recipes_dir.glob("**/*.toml"))]
        if category is not None:
            recipes = [recipe for recipe in recipes if recipe.category == category]
        return sorted(recipes, key=lambda recipe: recipe.id)

    def get(self, recipe_id: str) -> Recipe:
        for recipe in self.list():
            if recipe.id == recipe_id:
                return recipe
        available = ", ".join(recipe.id for recipe in self.list()) or "none"
        raise KeyError(f"Unknown recipe '{recipe_id}'. Available recipes: {available}")
