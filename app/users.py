"""User accounts: who may sign in (auth.py). There are no roles: whoever signs in may do
anything, adding and deleting accounts included. Passwords are kept as werkzeug's scrypt
hashes. Flask-free, so the command line below can use it.

From the server, for a forgotten password or a lockout (the app needn't be stopped):
  docker compose exec chatbot python -m app.users list
  docker compose exec chatbot python -m app.users add NAME        asks for a password
  docker compose exec chatbot python -m app.users password NAME   a new one; signs NAME out everywhere
  docker compose exec chatbot python -m app.users delete NAME
  docker compose exec chatbot python -m app.users setup-code      while there is no account yet
A password can also come on stdin (--password-stdin), for scripts."""
import argparse
import getpass
import re
import secrets
import sys

from werkzeug.security import check_password_hash, generate_password_hash

from . import db

NAME_CHARS = 40
MIN_PASSWORD, MAX_PASSWORD = 8, 200  # scrypt of a megabyte would hold a thread up for nothing
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*")
# Checked against when the name is unknown, so a wrong name takes as long as a wrong password
# and the time a login takes doesn't say which names exist.
_DUMMY = generate_password_hash(secrets.token_hex(16))


class Invalid(ValueError):
    """A name or password that can't be used; its message says why, for the user."""


class Taken(Invalid):
    pass


def check_name(name):
    name = name.strip() if isinstance(name, str) else ""
    if not name:
        raise Invalid("Choose a user name")
    if len(name) > NAME_CHARS or not _NAME.fullmatch(name):
        raise Invalid(f"A user name is up to {NAME_CHARS} letters, digits and . _ @ -, starting with a letter or digit")
    return name


def check_password(password):
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise Invalid(f"A password needs at least {MIN_PASSWORD} characters")
    if len(password) > MAX_PASSWORD:
        raise Invalid(f"A password is {MAX_PASSWORD} characters at most")
    return password


def _hash(password):
    return generate_password_hash(check_password(password))


def add(name, password, first=False):
    """A new account; its id. first: only while there is none (the setup page), else None."""
    name = check_name(name)
    hashed = _hash(password)
    try:
        return db.add_user(name, hashed, first=first)
    except db.IntegrityError:
        raise Taken(f"There is already a user called {name}") from None


def change(user_id, name=None, password=None):
    """Rename an account and/or set its password (which signs it out everywhere). The account
    as it now is, or None if there is none."""
    name = None if name is None else check_name(name)
    hashed = None if password is None else _hash(password)
    try:
        return db.update_user(user_id, name, hashed)
    except db.IntegrityError:
        raise Taken(f"There is already a user called {name}") from None


def verify(name, password):
    """The account these sign in, or None. As long either way (_DUMMY)."""
    user = db.find_user(name.strip()) if isinstance(name, str) and name.strip() else None
    ok = check_password_hash(user["password_hash"] if user else _DUMMY, password if isinstance(password, str) else "")
    return user if user and ok else None


def public(user):
    return {k: user[k] for k in ("id", "username", "created_at", "last_login_at")}


def setup_code():
    """While there is no account, the one-time code the setup page asks for, so whoever first
    reaches the page over the network can't make the first account: the app writes it in its
    log at start. Kept in the database (settings), so it stays the same across restarts and
    this command line can say it too; dropped when the first account is made."""
    return db.setting("setup_code", lambda: "-".join(secrets.token_hex(3) for _ in range(3)))


def forget_setup_code():
    with db.connect() as con:
        con.execute("DELETE FROM settings WHERE key = 'setup_code'")


# ---- command line ----

def _password(args, prompt="Password: "):
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass(prompt)
    if getpass.getpass("Again: ") != first:
        raise Invalid("The two passwords differ")
    return first


def _named(name):
    user = db.find_user(name)
    if not user:
        raise Invalid(f"No user called {name}")
    return user


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.users", description="Manage who may sign in to the library.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the accounts")
    for cmd, text in (("add", "a new account"), ("password", "a new password (signs the user out everywhere)")):
        s = sub.add_parser(cmd, help=text)
        s.add_argument("name")
        s.add_argument("--password-stdin", action="store_true", help="read the password from stdin")
    sub.add_parser("delete", help="delete an account").add_argument("name")
    sub.add_parser("setup-code", help="the code the setup page asks for, while there is no account")
    args = p.parse_args(argv)
    try:
        if args.cmd == "list":
            users = db.list_users()
            for u in users:
                print(f"{u['username']:<{NAME_CHARS}}  added {u['created_at'][:16]}  last sign-in "
                      f"{(u['last_login_at'] or 'never')[:16]}")
            if not users:
                print("No accounts yet: open the app to make the first, or use add.")
        elif args.cmd == "add":
            name = check_name(args.name)
            add(name, _password(args))
            forget_setup_code()
            print(f"Added {name}.")
        elif args.cmd == "password":
            user = _named(args.name)
            change(user["id"], password=_password(args, f"New password for {user['username']}: "))
            print(f"Password changed; {user['username']} is signed out everywhere.")
        elif args.cmd == "delete":
            user = _named(args.name)
            db.delete_user(user["id"])
            print(f"Deleted {user['username']}.")
        elif args.cmd == "setup-code":
            if db.user_count():
                raise Invalid("There are accounts already: sign in, or use add or password")
            print(setup_code())
    except Invalid as e:
        print(e, file=sys.stderr)
        return 1
    except db.LastUser:
        print("That is the only account left: add another first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
