"""Shared toolkit for the 16 projects: app factory, UI block builders, AutoResearch hill-climber."""
from .blocks import *  # noqa: F401,F403
from .server import make_app, load_local, clean, save_json, load_json, load_or_train  # noqa: F401
from .autoresearch import hill_climb, ar_blocks  # noqa: F401
