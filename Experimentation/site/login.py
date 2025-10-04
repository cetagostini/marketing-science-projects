import os
from typing import Dict

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for
from storage import load_experiments

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = lambda: None

load_dotenv()

users_str = os.getenv("APP_USERS", "")
USERS: Dict[str, str] = {}
for pair in users_str.split(","):
    if ":" in pair:
        username, password = pair.split(":", 1)
        USERS[username.strip()] = password.strip()

login_bp = Blueprint("login", __name__, template_folder="templates")


@login_bp.route("/login", methods=["GET", "POST"])
def login() -> str:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["user"] = username
            session["boot_token"] = current_app.config.get("BOOT_TOKEN")
            experiments = load_experiments(username)
            session.setdefault("experiments", {})[username] = experiments
            next_url = request.args.get("next") or "/"
            return redirect(next_url)
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@login_bp.route("/logout")
def logout() -> str:
    session.clear()
    return redirect(url_for("login.login"))
