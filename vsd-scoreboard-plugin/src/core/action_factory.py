import importlib
import inspect
import os
import sys
import pkgutil
from src.core.action import Action
from src.core.logger import Logger

# Explicit list of action modules — required when running as a PyInstaller bundle
# because pkgutil.iter_modules cannot enumerate modules from a frozen archive.
_KNOWN_ACTION_MODULES = [
    "src.actions.info_actions",
    "src.actions.score_actions",
    "src.actions.period_actions",
    "src.actions.exclusion_actions",
    "src.actions.timeout_actions",
    "src.actions.manup_actions",
    "src.actions.game_actions",
    "src.actions.replay_actions",
]


class ActionFactory:
    _registry: dict = {}   # action_name -> Action subclass

    @classmethod
    def register(cls, name: str, action_class):
        cls._registry[name] = action_class
        Logger.debug(f"Registered action: {name} -> {action_class.__name__}")

    @classmethod
    def create(cls, action_uuid: str, context: str, settings: dict, plugin):
        name = action_uuid.split(".")[-1]
        klass = cls._registry.get(name)
        if klass is None:
            Logger.error(f"No action registered for UUID suffix '{name}'")
            return None
        return klass(action_uuid, context, settings, plugin)

    @classmethod
    def scan_and_register(cls, package_name: str = "src.actions"):
        """Auto-discover every Action subclass inside the given package.

        When running as a frozen PyInstaller binary, pkgutil.iter_modules
        cannot enumerate the bundle archive, so we fall back to the explicit
        _KNOWN_ACTION_MODULES list instead.
        """
        frozen = getattr(sys, "frozen", False)
        if frozen:
            module_names = _KNOWN_ACTION_MODULES
        else:
            try:
                package = importlib.import_module(package_name)
                pkg_path = os.path.dirname(package.__file__)
                module_names = [
                    f"{package_name}.{mn}"
                    for _, mn, _ in pkgutil.iter_modules([pkg_path])
                ]
            except Exception as e:
                Logger.error(f"ActionFactory scan failed: {e}")
                return

        for full_name in module_names:
            try:
                mod = importlib.import_module(full_name)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        inspect.isclass(attr)
                        and issubclass(attr, Action)
                        and attr is not Action
                        and not attr_name.startswith("_")
                    ):
                        key = attr_name.lower()
                        if key.endswith("action"):
                            key = key[: -len("action")]
                        cls.register(key, attr)
            except Exception as e:
                Logger.error(f"Failed to import actions module '{full_name}': {e}")
