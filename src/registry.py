"""
src.registry — the one pattern that makes the whole thesis pluggable.

THE PROBLEM
-----------
The thesis runs ~30 experiments across four independent axes::

    backbone   ×   dataset   ×   objective   ×   metric
   (blip2,        (scienceqa,    (generative,    (mc_accuracy,
    qwen2_vl,      chartqa,       explanation,     rouge_l,
    paligemma)     docvqa, ...)   contrastive)     faithfulness)

If every experiment hard-codes ``Blip2ForConditionalGeneration`` and
``load_dataset("derek-thomas/ScienceQA")``, then adding Qwen2-VL or ChartQA
means editing the training loop. That does not scale to 30 runs.

THE FIX: a name → class lookup table you register into.

    BACKBONES = Registry("backbone")

    @BACKBONES.register("blip2")          # decorator, runs at import time
    class Blip2Wrapper(BaseVLMWrapper): ...

    wrapper = BACKBONES.build("blip2", cfg)   # later, by string, from a YAML file

Now a *config file* picks the backbone::

    model:
      name: blip2          # ← this string is the registry key

…and the training loop never names a concrete class. To add PaliGemma you
write one new file with one ``@BACKBONES.register("paligemma")`` decorator and
it is instantly selectable from YAML. Nothing else changes.

VISUAL
------
::

    Registry("backbone")  ─ a dict with a decorator bolted on ─

        register("blip2")     ┌──────────────────────────────────┐
        register("qwen2_vl")  │  "blip2"    → Blip2Wrapper        │
        register("paligemma") │  "qwen2_vl" → Qwen2VLWrapper      │
                              │  "paligemma"→ PaliGemmaWrapper    │
                              └──────────────────────────────────┘
                                         │
              build("qwen2_vl", cfg) ────┘  → Qwen2VLWrapper(cfg) instance

The four registries (BACKBONES, DATASETS, OBJECTIVES, METRICS) live next to the
code they index; see each subpackage's ``__init__``.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A typed, self-documenting name → class table with a decorator API.

    Generic over ``T`` so editors know what ``build`` returns: a
    ``Registry[BaseVLMWrapper]`` builds ``BaseVLMWrapper`` instances.

    Example
    -------
    >>> ANIMALS: Registry[str] = Registry("animal")
    >>> @ANIMALS.register("dog")
    ... class Dog:
    ...     def speak(self): return "woof"
    >>> ANIMALS.available()
    ['dog']
    >>> ANIMALS.build("dog").speak()
    'woof'
    >>> "dog" in ANIMALS
    True
    """

    def __init__(self, name: str) -> None:
        #: human-readable axis name, only used to make error messages legible.
        self.name = name
        self._table: dict[str, type[T]] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator that adds a class to the table under ``key``.

        If ``key`` is omitted the class's own ``__name__`` is used. Returns the
        class unchanged, so the decorator is transparent::

            @BACKBONES.register("blip2")
            class Blip2Wrapper(BaseVLMWrapper): ...   # still importable as Blip2Wrapper
        """

        def decorator(cls: type[T]) -> type[T]:
            resolved = key or cls.__name__
            if resolved in self._table:
                raise KeyError(
                    f"{self.name!r} registry already has a {resolved!r} "
                    f"(existing={self._table[resolved].__name__}). "
                    "Registry keys must be unique."
                )
            self._table[resolved] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Return the registered class for ``key`` (not an instance).

        Raises ``KeyError`` with the list of valid keys — so a typo in a YAML
        file fails loudly with ``"unknown backbone 'blpi2'. Available: ..."``
        instead of a cryptic ``None`` later.
        """
        if key not in self._table:
            raise KeyError(
                f"unknown {self.name} {key!r}. "
                f"Available: {self.available()}. "
                f"(Did you forget to import the module that registers it?)"
            )
        return self._table[key]

    def build(self, key: str, *args, **kwargs) -> T:
        """``get(key)`` then construct it: ``Registry.build('blip2', cfg)``."""
        return self.get(key)(*args, **kwargs)

    def available(self) -> list[str]:
        """Sorted list of registered keys — handy in error messages and ``--help``."""
        return sorted(self._table)

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={self.available()})"
