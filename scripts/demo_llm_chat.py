#!/usr/bin/env python3
"""
Démonstration terminal : même réponse JSON que POST /reporting/submit puis (optionnel) LLM via LiteLLM.

Prérequis — **remplacer** par vos vrais identifiants PostgreSQL :
  export DATABASE_URL='postgresql://MON_LOGIN_POSTGRES:MON_MOTDEPASSE@localhost:5432/municipall'
  export LITELLM_API_KEY='<clé>'   # sauf avec --submit-only ou --no-llm

Exemple (retour compact identique au curl API) :

  python scripts/demo_llm_chat.py --submit-only \\
    -m "Nid de poule rue de la Paix"

  {"report_id":"<uuid>","status":"Open", ...}

Interactif :

  python scripts/demo_llm_chat.py

Options :
  --no-llm, --submit-only  → pas d'appel LLM ; pas besoin de LITELLM_API_KEY
  -q, --quiet              → aucun bandeau / label ; stdout = uniquement ligne(s) JSON
  -v, --verbose            → après le JSON compact, bloc JSON enrichi (analysis, etc.)
  --user-id UUID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from urllib.parse import urlparse, unquote


def _validate_database_url(raw: str) -> int | None:
    """Retourne 1 si l'URL manifestement invalide, sinon None."""

    stripped = raw.strip()
    ellipsis = "\u2026"

    if ellipsis in stripped:
        print(
            "Erreur : DATABASE_URL contient des points de suspension Unicode, ce n'est pas un hôte valide.\n"
            "Utilisez par ex. : export DATABASE_URL='postgresql://USER:PASSWORD@localhost:5432/municipall'\n"
            "(remplacez USER et PASSWORD par vos identifiants PostgreSQL réels.)",
            file=sys.stderr,
        )
        return 1

    normalized = stripped.replace("postgresql+psycopg://", "postgresql://", 1)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").strip()
    if not host:
        print(
            "Erreur : DATABASE_URL sans nom d'hôte après @ (ex. @localhost:5432/ma_base).",
            file=sys.stderr,
        )
        return 1

    raw_user = unquote(parsed.username or "").strip()
    raw_pass = unquote(parsed.password or "").strip()
    u_lower = raw_user.lower()
    p_lower = raw_pass.lower()
    if u_lower == "ton_user" or raw_user == "UTILISATEUR" or raw_user == "USER":
        print(
            "Erreur : le nom d'utilisateur PostgreSQL est encore un **exemple de la doc**, pas votre vrai compte.\n"
            "Postgres cherchera un rôle nommé exactement comme dans l'URL — il n'existe pas.\n\n"
            "Exemples de vrais noms sous macOS/Homebrew : souvent votre login (`whoami`) ou `postgres`.\n"
            "Pour lister ou créer un rôle :\n"
            "  psql -U postgres -h localhost -d postgres -c '\\du'\n"
            "Ensuite :\n"
            "  export DATABASE_URL='postgresql://<vrai_login>:<vrai_mdp>@localhost:5432/municipall'",
            file=sys.stderr,
        )
        return 1
    if p_lower == "ton_motdepasse" or raw_pass == "MOTDEPASSE" or raw_pass == "PASSWORD":
        print(
            "Erreur : le mot de passe dans DATABASE_URL est encore un **placeholder**.\n"
            "Remplacez-le par le mot de passe réel du rôle PostgreSQL concerné.",
            file=sys.stderr,
        )
        return 1
    return None


def _as_api_submit_json(r: dict) -> dict:
    """Même forme que reporting_routes.SubmitOut / réponse curl."""
    return {
        "report_id": r["report_id"],
        "status": r["status"],
        "category": r["category"],
        "municipal_service": r["municipal_service"],
        "sentiment_score": float(r["sentiment_score"]),
        "is_spam": bool(r["is_spam"]),
        "duplicate_of_id": r.get("duplicate_of_id"),
    }


def _log(msg: str, *, quiet: bool, file=sys.stderr) -> None:
    if not quiet:
        print(msg, file=file)


def main() -> int:
    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print(
            "Erreur : définissez DATABASE_URL pour insérer les signalements.",
            file=sys.stderr,
        )
        return 1

    rc = _validate_database_url(db_url)
    if rc is not None:
        return rc

    parser = argparse.ArgumentParser(
        description="Démo : JSON type /reporting/submit + chat LLM optionnel"
    )
    parser.add_argument(
        "-m",
        "--message",
        help="Un seul message puis quitter",
    )
    parser.add_argument(
        "--user-id",
        dest="user_id",
        default=os.environ.get("DEMO_USER_ID", "").strip() or None,
        help="UUID citoyen (sinon aléatoire)",
    )
    parser.add_argument(
        "--no-llm",
        "--submit-only",
        dest="submit_only",
        action="store_true",
        help="N'appelle pas le LLM : affiche seulement le JSON signalement (comme curl).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Stdout = lignes JSON seules (bandeaux sur stderr seulement en cas d'erreur).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Après le JSON compact : JSON étendu (analysis, duplicate_check, router…).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from municipal.llm_client import chat_completion, llm_configured
        from municipal.pipeline import submit_report
    except ImportError as e:
        print(f"Erreur d'import : {e}", file=sys.stderr)
        return 1

    if not args.submit_only and not llm_configured():
        print(
            "Erreur : LITELLM_API_KEY absente. Utilisez --submit-only ou --no-llm pour ne tester que le signalement.",
            file=sys.stderr,
        )
        return 1

    user_id = args.user_id or str(uuid.uuid4())

    sys_msg = (
        "Tu es l'assistant de démonstration Municip'All pour opérateurs et élus.\n"
        "À chaque message un signalement est en base : tu reçois son JSON officiel (/reporting/submit)\n"
        "puis les champs métier enrichis pour la synthèse.\n"
        "Réponds en français, sobre, 4 phrases max :\n"
        "- statut, catégorie, service ciblé\n"
        "- sentiment / urgence / spam éventuel\n"
        "- phrase type pour répondre au citoyen (sans inventer de faits)."
    )

    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_msg}] if not args.submit_only else []

    _log("———— Démo Municip'All ————", quiet=args.quiet)
    _log(f"DATABASE_URL défini — user_id = {user_id}", quiet=args.quiet)
    if args.submit_only:
        _log("Mode --submit-only (pas d'appel LLM).", quiet=args.quiet)
    _log("", quiet=args.quiet)

    lines: list[str] = []
    if args.message:
        lines = [args.message.strip()]
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            s = line.strip()
            if s:
                lines.append(s)
    else:
        print("Tapez un message citoyen (ligne vide ou Ctrl+D pour quitter).\n", file=sys.stderr)
        try:
            while True:
                line = input("Vous> ").strip()
                if not line:
                    break
                lines.append(line)
        except EOFError:
            pass

        if not lines:
            _log("Au revoir.", quiet=args.quiet)
            return 0

    if not lines:
        print("Aucun message : utilisez -m et des guillemets ou envoyez du texte sur stdin.", file=sys.stderr)
        return 1

    for idx, raw in enumerate(lines, start=1):
        if not args.quiet and len(lines) > 1:
            print(f"\n─── Message #{idx} ───", file=sys.stderr)
            print(f"Citoyen : {raw}\n", file=sys.stderr)

        try:
            result = submit_report(user_id, raw)
        except Exception as e:
            msg = str(e).lower()
            print(f"Erreur pipeline / base : {e}", file=sys.stderr)
            if (
                "nodename nor servname" in msg
                or "failed to resolve host" in msg
                or "could not translate host name" in msg
            ):
                print(
                    "\nAstuce : vérifiez DATABASE_URL (@localhost ou @127.0.0.1).",
                    file=sys.stderr,
                )
            if 'role "' in msg and "does not exist" in msg:
                print(
                    "\nAstuce : PostgreSQL refuse le nom d'utilisateur dans l'URL : ce rôle n'existe pas.\n"
                    "Vérifiez avec votre admin ou : psql postgres -c '\\du'.",
                    file=sys.stderr,
                )
            return 1

        api_line = _as_api_submit_json(result)
        print(json.dumps(api_line, ensure_ascii=False), flush=True)

        if args.verbose:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)

        if args.submit_only:
            continue

        msgs.append(
            {
                "role": "user",
                "content": (
                    "Signalement traité.\nRéponse officielle POST /reporting/submit :\n"
                    + json.dumps(api_line, ensure_ascii=False)
                    + "\n\nDonnées complètes (interne démo) :\n"
                    + json.dumps(result, ensure_ascii=False, default=str)
                ),
            },
        )

        try:
            answer = chat_completion(msgs, temperature=0.25)
        except Exception as e:
            print(f"Erreur LLM : {e}", file=sys.stderr)
            msgs.pop()
            return 1

        msgs.append({"role": "assistant", "content": answer})

        _log("", quiet=args.quiet)
        _log("── Réponse LLM ──", quiet=args.quiet)
        print(answer, flush=True)
        _log("", quiet=args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
