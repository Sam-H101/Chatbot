"""Signing in with a user account (users.py). Every page and API call needs the signed session
cookie /login gives, except the sign-in and setup pages themselves, which have no assets of
their own. Whoever is signed in may do anything. With no account yet, every page goes to
/setup, which makes the first one and asks for the one-time setup code the app writes in its
log, so whoever first reaches it over the network can't claim the library. Background jobs
don't pass through here."""
import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import time
from datetime import timedelta
from urllib.parse import urlsplit

from flask import Blueprint, current_app, g, jsonify, redirect, render_template, request, session, url_for
from flask.sessions import SecureCookieSessionInterface

from . import db, users

log = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)

REQUIRED = True  # tests of everything else switch it off
SESSION_DAYS = int(os.environ.get("SESSION_DAYS", "30"))
# Failed sign-ins (and setup codes) per client address: after FAILS within WINDOW seconds, every
# further try waits until LOCK seconds after the last failure, the right password included.
FAILS, WINDOW, LOCK = 5, 600, 60
# A state-changing API call must carry this header. A page on another site can make a browser
# send a form or a plain POST here, but not with a header of its own (that needs CORS, which
# this server never grants), so the header proves the request came from the app's own page.
HEADER = "X-Requested-With"
OPEN = {"auth.login_page", "auth.login", "auth.logout", "auth.setup_page", "auth.first_account"}  # no sign-in needed
_fails, _fails_lock, _key_lock = {}, threading.Lock(), threading.Lock()
_now = time.monotonic
_announced = []


def setup(app):
    """Hook the sign-in into the app. Its before_request runs after those already registered."""
    app.session_interface = Sessions()
    app.config.update(SESSION_COOKIE_NAME="docchat_session", SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(days=SESSION_DAYS),
                      # The cookie lasts SESSION_DAYS from signing in. Renewed with every request, a
                      # poll answered just after Log out would bring back the session it ended.
                      SESSION_REFRESH_EACH_REQUEST=False)
    app.register_blueprint(bp)
    app.before_request(gate)
    if os.environ.get("APP_PASSWORD"):
        log.warning("APP_PASSWORD is no longer used: everyone signs in with an account of their own (README, Sign-in)")


def announce():
    """With no account yet, say so in the log with the setup code, once per process."""
    if _announced or not REQUIRED:
        return
    if db.user_count():
        _announced.append(True)
        return
    _announced.append(True)
    log.warning("No user accounts yet: open the app to make the first one, with this setup code: %s", users.setup_code())


def secret_key():
    """SECRET_KEY, or else a random key made once and kept in the database (db.setting), so a
    restart doesn't sign everyone out. Deleting it (DELETE FROM settings WHERE key='secret_key')
    signs everyone out at the next restart."""
    key = os.environ.get("SECRET_KEY", "")
    if key:
        return key
    return db.setting("secret_key", lambda: secrets.token_hex(32))


class Sessions(SecureCookieSessionInterface):
    """Flask's signed cookie, with its key loaded when first needed. With sign-in off (tests)
    there is no session at all: a cookie is neither read nor set."""

    def get_signing_serializer(self, app):
        if not REQUIRED:
            return None
        if not app.secret_key:
            with _key_lock:
                if not app.secret_key:
                    app.secret_key = secret_key()
        return super().get_signing_serializer(app)


def _tag(user):
    """What a signed-in session holds beside the account's id. It depends on the account's
    epoch, so a new password signs it out everywhere; keyed with the secret, so it can't be
    made up for another account (the cookie is signed, not encrypted)."""
    key = current_app.secret_key
    key = key.encode() if isinstance(key, str) else key
    return hmac.new(key, f"user\0{user['id']}\0{user['epoch']}".encode(), hashlib.sha256).hexdigest()


def current_user():
    """The account this request is signed in as, or None. Read once per request."""
    if "user" not in g:
        uid = session.get("uid")
        user = db.get_user(uid) if isinstance(uid, int) else None
        g.user = user if user and hmac.compare_digest(str(session.get("tag", "")), _tag(user)) else None
    return g.user


def _sign_in(user):
    session.clear()
    session.permanent = True  # SESSION_DAYS, rather than until the browser closes
    session["uid"], session["tag"] = user["id"], _tag(user)
    g.user = user


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
    """Where to go after signing in: a path on this site, nothing else (no //host, no scheme),
    with the #fragment the sign-in page adds (an original's #page=5), which the server never sees.
    Browsers drop tabs and line breaks from a URL, so "/\\t/host" would become "//host": a next
    with any whitespace or control character is refused too (one made by gate() has none)."""
    if (not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw
            or any(c <= " " or c == "\x7f" for c in raw)):
        return "/"
    parts = urlsplit(raw)
    return "/" if parts.scheme or parts.netloc else raw


def _here():
    """The page asked for, as a sign-in's `next`."""
    query = request.query_string.decode("latin-1")
    return request.path + ("?" + query if query else "")


def gate():
    """Without a sign-in, pages go to /login (/setup while there is no account) and API calls
    get 401 (the page then goes to /login). A browser opening an API link itself (an original
    file, from a bookmark) is sent to /login too, and back after. Signed in, a state-changing
    API call must carry HEADER."""
    if not REQUIRED or request.endpoint in OPEN:
        return None
    api = request.path.startswith("/api/")
    if not current_user():
        if api and not (request.method in ("GET", "HEAD") and request.headers.get("Sec-Fetch-Mode") == "navigate"):
            return jsonify(error="Sign-in required", login=True), 401
        here = _here()
        page = "auth.login_page" if db.user_count() else "auth.setup_page"
        return redirect(url_for(page, **({"next": here} if here != "/" else {})))
    if api and request.method not in ("GET", "HEAD", "OPTIONS") and not request.headers.get(HEADER):
        return jsonify(error=f"State-changing API calls need an {HEADER} header (any value)"), 403
    return None


def _page(next_url, error=None, status=200, wait=0, ended=False, setup=False, username=""):
    page = render_template("login.html", next_url=next_url, error=error, ended=ended, setup=setup, username=username,
                           min_password=users.MIN_PASSWORD, name_chars=users.NAME_CHARS)
    resp = current_app.make_response((page, status))
    resp.headers["Cache-Control"] = "no-store"
    if wait:
        resp.headers["Retry-After"] = str(wait)
    return resp


def _held_up(ip, now, next_url, **page):
    """The page saying to wait, if this address has failed too often lately; else None."""
    wait = math.ceil(_wait(ip, now))
    if wait:
        return _page(next_url, f"Too many failed tries. Try again in {wait} seconds.", 429, wait, **page)
    return None


@bp.get("/login")
def login_page():
    if not REQUIRED:
        return redirect("/")
    next_url = _next(request.args.get("next"))
    if current_user():
        return redirect(next_url)
    if not db.user_count():
        return redirect(url_for("auth.setup_page"))
    # ?ended=1: the app's page was sent here by a 401, so the session ended under it.
    return _page(next_url, ended=request.args.get("ended") == "1")


@bp.post("/login")
def login():
    """Sign in with a user name and password. Failures are counted per client address (_wait)."""
    if not REQUIRED:
        return redirect("/", 303)
    next_url, ip = _next(request.form.get("next")), request.remote_addr or "?"
    name = request.form.get("username", "")
    with _fails_lock:
        now = _now()
        if held := _held_up(ip, now, next_url, username=name):
            return held
    user = users.verify(name, request.form.get("password", ""))  # slow (scrypt): not under the lock
    with _fails_lock:
        if not user:
            _fail(ip, now)
            log.warning("Failed sign-in from %s", ip)
            wait = math.ceil(_wait(ip, now))
            if wait:
                return _page(next_url, f"Wrong user name or password. Too many tries: wait {wait} seconds.", 429, wait,
                             username=name)
            return _page(next_url, "Wrong user name or password.", 401, username=name)
        _fails.pop(ip, None)
    db.logged_in_now(user["id"])
    _sign_in(user)
    return redirect(next_url, 303)


@bp.get("/setup")
def setup_page():
    if not REQUIRED or db.user_count():
        return redirect(url_for("auth.login_page") if REQUIRED else "/")
    announce()
    return _page(_next(request.args.get("next")), setup=True)


@bp.post("/setup")
def first_account():
    """Make the first account, with the setup code from the log, and sign in as it."""
    if not REQUIRED or db.user_count():
        return redirect(url_for("auth.login_page") if REQUIRED else "/", 303)
    next_url, ip, form = _next(request.form.get("next")), request.remote_addr or "?", request.form
    name = form.get("username", "")
    with _fails_lock:
        now = _now()
        if held := _held_up(ip, now, next_url, setup=True, username=name):
            return held
        code = "".join(form.get("code", "").split()).lower()
        if not hmac.compare_digest(code.encode(), users.setup_code().encode()):
            _fail(ip, now)
            log.warning("Wrong setup code from %s", ip)
            return _page(next_url, "Wrong setup code: it is in the app's log (docker compose logs chatbot).", 401,
                         setup=True, username=name)
        _fails.pop(ip, None)
    if form.get("password", "") != form.get("password2", ""):
        return _page(next_url, "The two passwords differ.", 400, setup=True, username=name)
    try:
        user_id = users.add(name, form.get("password", ""), first=True)
    except users.Invalid as e:
        return _page(next_url, f"{e}.", 400, setup=True, username=name)
    if user_id is None:  # someone else made the first account meanwhile
        return redirect(url_for("auth.login_page"), 303)
    users.forget_setup_code()
    log.warning("First user account made: %s", users.check_name(name))
    db.logged_in_now(user_id)
    _sign_in(db.get_user(user_id))
    return redirect(next_url, 303)


@bp.post("/logout")
def logout():
    if not REQUIRED:  # no session to end
        return redirect("/", 303)
    session.clear()  # the cookie is deleted
    return redirect(url_for("auth.login_page"), 303)


@bp.get("/api/session")
def session_info():
    """{"login": true, "user": {id, username}} signed in; {"login": false, "user": null} with
    sign-in off. The page shows the account menu from it."""
    user = current_user() if REQUIRED else None
    return jsonify(login=REQUIRED, user=user and {"id": user["id"], "username": user["username"]})


# ---- managing accounts: anyone signed in may ----

def json_body():
    """The request's JSON object; {} for none or anything else (its fields are then missing)."""
    body = request.get_json(force=True, silent=True)
    return body if isinstance(body, dict) else {}


def _me():
    user = current_user() if REQUIRED else None
    return user and user["id"]


@bp.get("/api/users")
def user_list():
    me = _me()
    return jsonify([{**u, "you": u["id"] == me} for u in db.list_users()])


@bp.post("/api/users")
def user_add():
    body = json_body()
    try:
        user_id = users.add(body.get("username"), body.get("password"))
    except users.Taken as e:
        return jsonify(error=str(e)), 409
    except users.Invalid as e:
        return jsonify(error=str(e)), 400
    log.warning("User account %s added", users.check_name(body.get("username")))
    return jsonify({**users.public(db.get_user(user_id)), "you": False}), 201


@bp.patch("/api/users/<int:user_id>")
def user_edit(user_id):
    """{username?, password?}. A new password signs that account out everywhere; your own
    stays signed in here."""
    body = json_body()
    name, password = body.get("username"), body.get("password")
    if name is None and password is None:
        return jsonify(error="Nothing to change: give a username or a password"), 400
    try:
        user = users.change(user_id, name, password)
    except users.Taken as e:
        return jsonify(error=str(e)), 409
    except users.Invalid as e:
        return jsonify(error=str(e)), 400
    if not user:
        return jsonify(error="No such user (deleted meanwhile?)"), 404
    me = _me()
    if password is not None and user["id"] == me:
        _sign_in(user)  # the new epoch: this session goes on, the others end
    return jsonify({**users.public(user), "you": user["id"] == me})


@bp.delete("/api/users/<int:user_id>")
def user_delete(user_id):
    """Delete an account; its sessions end. Your own too (the page then goes to sign in), but
    never the last one."""
    try:
        if not db.delete_user(user_id):
            return jsonify(error="No such user (deleted meanwhile?)"), 404
    except db.LastUser:
        return jsonify(error="That is the only account: add another before deleting it"), 409
    if user_id == _me():
        session.clear()
    return "", 204
