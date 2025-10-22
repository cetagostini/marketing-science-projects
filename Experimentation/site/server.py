"""Shared Flask server configuration for the experimentation dashboard."""

from __future__ import annotations

import os

from flask import Flask
from openai import OpenAI

from login import login_bp


def _build_openai_client(api_key: str | None) -> OpenAI | None:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def create_server() -> Flask:
    server = Flask(__name__)
    server.secret_key = os.getenv("APP_SECRET", "dev-secret")
    server.register_blueprint(login_bp)
    server.config["BOOT_TOKEN"] = os.getenv("APP_BOOT_TOKEN") or os.urandom(16).hex()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    server.config["OPENAI_CLIENT"] = _build_openai_client(openai_api_key)

    return server


server = create_server()


