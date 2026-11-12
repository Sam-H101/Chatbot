"""An optional shared password (APP_PASSWORD). Unset or empty: no login, as before. Set: every
page and API call needs the signed session cookie /login gives, except the login page itself,
which has no assets of its own. Background jobs don't pass through here."""
import hashlib
import hmac
import logging
import math
import os
import secrets
import tempfile
import threading
import time
from datetime import timedelta
from urllib.parse import urlsplit

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from flask.sessions import SecureCookieSessionInterface

from . import db

log = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)

PASSWORD = os.environ.get("APP_PASSWORD", "")
SESSION_DAYS = int(os.environ.get("SESSION_DAYS", "30"))
# Failed logins per client address: after FAILS within WINDOW seconds, every further try waits
# until LOCK seconds after the last failure, the right password included.
FAILS, WINDOW, LOCK = 5, 600, 60
# A state-changing API call must carry this header. A page on another site can make a browser
# send a form or a plain POST here, but not with a header of its own (that needs CORS, which
# this server never grants), so the header proves the request came from the app's own page.
HEADER = "X-Requested-With"
OPEN = {"auth.login_page", "auth.login", "auth.logout"}  # reachable without logging in
_fails, _fails_lock, _key_lock = {}, threading.Lock(), threading.Lock()
_now = time.monotonic


def setup(app):
    """Hook the login into the app. Its before_request runs after those already registered."""
    app.session_interface = Sessions()
    app.config.update(SESSION_COOKIE_NAME="docchat_session", SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(days=SESSION_DAYS),
                      # The cookie lasts SESSION_DAYS from logging in. Renewed with every request, a
                      # poll answered just after Log out would bring back the session it ended.
                      SESSION_REFRESH_EACH_REQUEST=False)
    app.register_blueprint(bp)
    app.before_request(gate)
    # Said at start, so a password compose has emptied (a $ in .env read as a variable) shows in
    # the log. A warning, as that is what reaches the container's log.
    if PASSWORD:
        log.warning("Login on: APP_PASSWORD is set, every page and API call asks for it")
    else:
        log.warning("No login: APP_PASSWORD is unset or empty, so anyone who can reach the app can use it")


def secret_key():
    """SECRET_KEY, or else a random key made once and kept beside the database (mode 600), so a
    restart doesn't log everyone out. Deleting the file logs everyone out at the next restart."""
    key = os.environ.get("SECRET_KEY", "")
    if key:
        return key
    folder = os.path.dirname(db.DB_PATH) or "."
    path = os.path.join(folder, "secret_key")
    try:
        with open(path) as f:
            key = f.read().strip()
    except FileNotFoundError:
        key = ""
    if len(key) < 32:  # none yet, or a write cut short
        os.makedirs(folder, exist_ok=True)
        key = secrets.token_hex(32)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".secret_key.")  # mkstemp makes it mode 600
        with os.fdopen(fd, "w") as f:
            f.write(key)
        os.replace(tmp, path)
        log.info("Made a new session key in %s", path)
    return key


class Sessions(SecureCookieSessionInterface):
    """Flask's signed cookie, with its key loaded when first needed. Without a password there
    is no session at all: a cookie is neither read nor set."""

    def get_signing_serializer(self, app):
        if not PASSWORD:
            return None
        if not app.secret_key:
            with _key_lock:
                if not app.secret_key:
                    app.secret_key = secret_key()
        return super().get_signing_serializer(app)


def _tag():
    """What a logged-in session holds. It depends on the password, so changing APP_PASSWORD logs
    everyone out; keyed with the secret, so the cookie (signed, not encrypted) gives nothing away."""
    key = current_app.secret_key
    key = key.encode() if isinstance(key, str) else key
    return hmac.new(key, b"login\0" + PASSWORD.encode(), hashlib.sha256).hexdigest()


def logged_in():
    return hmac.compare_digest(str(session.get("login", "")), _tag())


def _matches(given):
    # Digests of both, so the comparison takes as long whatever the lengths.
    return hmac.compare_digest(hashlib.sha256(given.encode()).digest(), hashlib.sha256(PASSWORD.encode()).digest())


def _wait(ip, now):
    """Seconds this address must wait before its next try (0: it may try now). Caller holds _fails_lock."""
    fails = [t for t in _fails.get(ip, ()) if now - t < WINDOW]
    if fails:
        _fails[ip] = fails
    else:
        _fails.pop(ip, None)
    return max(0.0, LOCK - (now - fails[-1])) if len(fails) >= FAILS else 0.0


def _fail(ip, now):
    for other in [k for k, v in _fails.items() if now - v[-1] >= WINDOW]:  # addresses long gone
        del _fails[other]
    _fails.setdefault(ip, []).append(now)


def _next(raw):
    """Where to go after logging in: a path on this site, nothing else (no //host, no scheme),
    with the #fragment the login page adds (an original's #page=5), which the server never sees.
    Browsers drop tabs and line breaks from a URL, so "/\\t/host" would become "//host": a next
    with any whitespace or control character is refused too (one made by gate() has none)."""
    if (not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw
            or any(c <= " " or c == "\x7f" for c in raw)):
        return "/"
    parts = urlsplit(raw)
    return "/" if parts.scheme or parts.netloc else raw


def _here():
    """The page asked for, as a login's `next`."""
    query = request.query_string.decode("latin-1")
    return request.path + ("?" + query if query else "")


def gate():
    """Without a login, pages go to /login and API calls get 401 (the page then goes to /login).
    A browser opening an API link itself (an original file, from a bookmark) is sent to /login
    too, and back after. Logged in, a state-changing API call must carry HEADER."""
    if not PASSWORD or request.endpoint in OPEN:
        return None
    api = request.path.startswith("/api/")
    if not logged_in():
        if api and not (request.method in ("GET", "HEAD") and request.headers.get("Sec-Fetch-Mode") == "navigate"):
            return jsonify(error="Login required", login=True), 401
        here = _here()
        return redirect(url_for("auth.login_page", **({"next": here} if here != "/" else {})))
    if api and request.method not in ("GET", "HEAD", "OPTIONS") and not request.headers.get(HEADER):
        return jsonify(error=f"State-changing API calls need an {HEADER} header (any value)"), 403
    return None


def _page(next_url, error=None, status=200, wait=0, ended=False):
    page = render_template("login.html", next_url=next_url, error=error, ended=ended)
    resp = current_app.make_response((page, status))
    resp.headers["Cache-Control"] = "no-store"
    if wait:
        resp.headers["Retry-After"] = str(wait)
    return resp


@bp.get("/login")
def login_page():
    if not PASSWORD:
        return redirect("/")
    next_url = _next(request.args.get("next"))
    # ?ended=1: the app's page was sent here by a 401, so the session ended under it.
    return redirect(next_url) if logged_in() else _page(next_url, ended=request.args.get("ended") == "1")


@bp.post("/login")
def login():
    """Log in with the shared password. Wrong ones are counted per client address (_wait)."""
    if not PASSWORD:
        return redirect("/", 303)
    next_url, ip = _next(request.form.get("next")), request.remote_addr or "?"
    with _fails_lock:
        now = _now()
        wait = math.ceil(_wait(ip, now))
        if wait:
            return _page(next_url, f"Too many wrong passwords. Try again in {wait} seconds.", 429, wait)
        if not _matches(request.form.get("password", "")):
            _fail(ip, now)
            log.warning("Wrong password from %s", ip)
            wait = math.ceil(_wait(ip, now))
            if wait:
                return _page(next_url, f"Wrong password. Too many tries: wait {wait} seconds.", 429, wait)
            return _page(next_url, "Wrong password.", 401)
        _fails.pop(ip, None)
    session.clear()
    session.permanent = True  # SESSION_DAYS, rather than until the browser closes
    session["login"] = _tag()
    return redirect(next_url, 303)


@bp.post("/logout")
def logout():
    if not PASSWORD:  # no session to end
        return redirect("/", 303)
    session.clear()  # the cookie is deleted
    return redirect(url_for("auth.login_page"), 303)


@bp.get("/api/session")
def session_info():
    """{"login": true} when a password guards the app: the page then shows Log out."""
    return jsonify(login=bool(PASSWORD))
