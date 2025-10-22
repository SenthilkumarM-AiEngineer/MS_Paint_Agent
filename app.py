# app.py
import os
import json
import html
import pandas as pd
from datetime import datetime

import streamlit as st
from action import Action
from ner import NER
from control import Control

st.set_page_config(page_title="Intelligent MS Paint Agent", page_icon="🎨")

# -------------------- Constants --------------------
HIST_CSV = "data/history/chat_history.csv"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
STATE_CSV = "data/history/shapes_state.csv"  # for delete helpers

# -------------------- Storage helpers --------------------
def _ensure_csv_exists():
    """
    Ensure the history CSV exists with required columns.
    If it exists but lacks 'updated_at', add it and initialize from 'created_at'.
    """
    os.makedirs(os.path.dirname(HIST_CSV), exist_ok=True)

    if not os.path.exists(HIST_CSV):
        df = pd.DataFrame(columns=["id", "title", "created_at", "updated_at", "messages_json"])
        df.to_csv(HIST_CSV, index=False)
        return

    # Migrate old schema to include updated_at
    try:
        df = pd.read_csv(HIST_CSV)
    except Exception:
        df = pd.DataFrame(columns=["id", "title", "created_at", "updated_at", "messages_json"])

    changed = False
    if "updated_at" not in df.columns:
        df["updated_at"] = df.get("created_at", "")
        changed = True

    for col in ["id", "title", "created_at", "updated_at", "messages_json"]:
        if col not in df.columns:
            df[col] = "" if col != "messages_json" else "[]"
            changed = True

    if changed:
        df.to_csv(HIST_CSV, index=False)

def _load_all_chats():
    """Load all chat histories, sorted by most recent activity (updated_at desc)."""
    _ensure_csv_exists()
    if not os.path.exists(HIST_CSV):
        return []

    try:
        df = pd.read_csv(HIST_CSV)
    except Exception:
        return []

    if df.empty:
        return []

    # Normalize timestamp columns
    df["updated_at"] = df.get("updated_at", "").fillna("")
    df["updated_at_dt"] = pd.to_datetime(df["updated_at"], format=DATE_FMT, errors="coerce")
    df["updated_at_dt"] = df["updated_at_dt"].fillna(pd.Timestamp.min)

    # Sort by last activity (newest first)
    df = df.sort_values(by="updated_at_dt", ascending=False, na_position="last")

    chats = []
    for _, row in df.iterrows():
        try:
            chats.append({
                "id": str(row["id"]),
                "title": str(row.get("title", "")) if not pd.isna(row.get("title", "")) else "",
                "created_at": str(row.get("created_at", "")),
                "updated_at": str(row.get("updated_at", "")),
                "messages": json.loads(row.get("messages_json", "[]") or "[]")
            })
        except Exception:
            chats.append({
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")) if not pd.isna(row.get("title", "")) else "",
                "created_at": str(row.get("created_at", "")),
                "updated_at": str(row.get("updated_at", "")),
                "messages": []
            })
    return chats

def _save_or_update_chat(chat_id: str, messages: list, title: str, created_at: str):
    """
    Create/update a chat row. We ALWAYS bump updated_at to now on any save,
    so the sidebar list reflects recent activity.
    """
    if not chat_id:
        return

    _ensure_csv_exists()
    now_str = datetime.now().strftime(DATE_FMT)

    try:
        df = pd.read_csv(HIST_CSV)
    except Exception:
        df = pd.DataFrame(columns=["id", "title", "created_at", "updated_at", "messages_json"])

    for col in ["id", "title", "created_at", "updated_at", "messages_json"]:
        if col not in df.columns:
            df[col] = ""

    messages_json = json.dumps(messages, ensure_ascii=False)

    if (df["id"] == chat_id).any():
        old_created = created_at
        if not old_created:
            try:
                old_created = str(df.loc[df["id"] == chat_id, "created_at"].iloc[0])
            except Exception:
                old_created = now_str

        df.loc[df["id"] == chat_id, ["title", "created_at", "updated_at", "messages_json"]] = [
            title, old_created, now_str, messages_json
        ]
    else:
        df = pd.concat([df, pd.DataFrame([{
            "id": chat_id,
            "title": title,
            "created_at": created_at or now_str,
            "updated_at": now_str,
            "messages_json": messages_json
        }])], ignore_index=True)

    df.to_csv(HIST_CSV, index=False)

def _truncate(s: str, n: int = 48) -> str:
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

def _first_word(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "(untitled)"
    return s.split()[0]

def _next_chat_id():
    chats = _load_all_chats()
    maxn = 0
    for c in chats:
        cid = str(c.get("id", ""))
        if cid.startswith("chat-"):
            try:
                n = int(cid.split("-", 1)[1])
                if n > maxn:
                    maxn = n
            except Exception:
                pass
    return f"chat-{maxn + 1}"

# ------ Chat admin helpers: delete (PNG + CSV + shapes_state) ------
def _delete_chat_everywhere(chat_id: str):
    try:
        Control.delete_session(chat_id)
    except Exception:
        pass
    try:
        from control import SAVE_ROOT as _SR
        fpath = os.path.join(_SR, f"{chat_id}.png")
        if os.path.exists(fpath):
            os.remove(fpath)
    except Exception:
        pass
    _ensure_csv_exists()
    try:
        df = pd.read_csv(HIST_CSV)
        df = df[df["id"] != chat_id]
        df.to_csv(HIST_CSV, index=False)
    except Exception:
        pass
    if os.path.exists(STATE_CSV):
        try:
            sdf = pd.read_csv(STATE_CSV)
            if "session" in sdf.columns:
                sdf = sdf[sdf["session"] != chat_id]
                sdf.to_csv(STATE_CSV, index=False)
        except Exception:
            pass

# -------------------- Session init --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "session_name" not in st.session_state:
    st.session_state.session_name = ""   # empty until first prompt
if "created_at" not in st.session_state:
    st.session_state.created_at = ""     # set on first prompt
if "title" not in st.session_state:
    st.session_state.title = ""
if "did_first_write" not in st.session_state:
    st.session_state.did_first_write = False
if "did_first_write_rerun_done" not in st.session_state:
    st.session_state.did_first_write_rerun_done = False

# -------------------- Helpers: model wiring & normalization --------------------
ANCHOR_WORDS = {"top","bottom","left","right","center","middle","centre","down","up"}

def _resolve_position_from_list(pos_list, default_pos: str = "center") -> str:
    """
    Convert an array like ["Top","Right"] into "top_right".
    Handle synonyms and order-insensitive corners.
    If empty or invalid, use internal default "center".
    """
    if not pos_list:
        return default_pos

    items = [str(p).strip().lower().replace("-", "_") for p in pos_list if p]
    mapped = []
    for p in items:
        if p in ("down",): p = "bottom"
        elif p in ("up",): p = "top"
        elif p in ("centre", "middle"): p = "center"
        mapped.append(p)

    items = [p for p in mapped if p in ANCHOR_WORDS]

    if not items:
        return default_pos
    if "center" in items:
        return "center"

    vertical = None
    horizontal = None
    for p in items:
        if p in ("top","bottom"):
            vertical = p
        elif p in ("left","right"):
            horizontal = p

    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    if vertical:
        return vertical
    if horizontal:
        return horizontal
    return default_pos

def _normalize_size(size_value: str | None, default_size: str = "medium") -> str:
    """
    Map NER size synonyms to one of: small / medium / big
    'large' is treated as 'big'
    """
    if not size_value:
        return default_size
    s = str(size_value).strip().lower()
    if s == "large":
        s = "big"
    return s if s in {"small","medium","big"} else default_size

def _normalize_size_opt(size_value: str | None) -> str | None:
    """
    Size normalizer for FILTERS (delete). If size is not provided or invalid,
    return None so we don't accidentally force a default like 'medium'.
    """
    if not size_value:
        return None
    s = str(size_value).strip().lower()
    if s == "large":
        s = "big"
    return s if s in {"small","medium","big"} else None

def _extract_action_entities(user_text: str) -> dict:
    """
    Use the AI models to predict action + entities.
    Ensure position is always a list (possibly empty).
    """
    act = (Action.predict(user_text) or "").strip().lower()
    if act in ("draw","delete"):
        ents = NER.predict(user_text) or {}
        # coerce position to list
        if isinstance(ents.get("position"), list):
            pass
        elif ents.get("position") is None:
            ents["position"] = []
        else:
            ents["position"] = [ents["position"]]
        return {"action": act, "entities": ents}
    elif act in ("redo","unknown"):
        return {"action": act, "entities": {}}
    else:
        return {"action": "unknown", "entities": {}}

def say(msg: str):
    """Append assistant message to history and persist; rendering happens in the transcript loop."""
    st.session_state.chat.append({"role": "assistant", "content": msg})
    _save_or_update_chat(
        st.session_state.session_name,
        st.session_state.chat,
        st.session_state.title,
        st.session_state.created_at
    )

# -------------------- Header --------------------
st.title("🎨 Intelligent MS Paint Agent")

# ==================== Input & flow (handled BEFORE sidebar) ====================
user_text = st.chat_input("Try: draw a large house at top left · delete previous · delete at bottom_right · redo")

if user_text:
    is_first_message_of_blank_session = not bool(st.session_state.session_name)

    if is_first_message_of_blank_session:
        st.session_state.session_name = _next_chat_id()
        now_str = datetime.now().strftime(DATE_FMT)
        st.session_state.created_at = now_str
        st.session_state.title = _first_word(user_text)

    # Append user message (no immediate echo; transcript will render below)
    st.session_state.chat.append({"role": "user", "content": user_text})
    _save_or_update_chat(
        st.session_state.session_name,
        st.session_state.chat,
        st.session_state.title,
        st.session_state.created_at
    )
    if is_first_message_of_blank_session:
        st.session_state.did_first_write = True

    model_out = _extract_action_entities(user_text)
    action = model_out["action"]
    ents = model_out.get("entities", {}) or {}

    # Resolve from AI (internal fallbacks only)
    resolved_pos = _resolve_position_from_list(ents.get("position", []), "center")
    resolved_size = _normalize_size(ents.get("size"), "medium")   # used for draw defaults
    size_filter   = _normalize_size_opt(ents.get("size"))         # used for delete filters
    shape = (ents.get("shape") or "").strip().lower()
    previous_flag = (ents.get("previous") or "").strip().lower()

    if action == "redo":
        res = Control.redo_previous_delete(st.session_state.session_name)
        if res.get("ok"):
            say("✅ Redid last delete.")
        else:
            reason = res.get("reason")
            msg_map = {
                "process_closed": "⚠️ Redo unavailable: original Paint session is not running.",
                "window_closed": "⚠️ Redo unavailable: the session's Paint window is not open.",
                "nothing_to_redo": "⚠️ Nothing to redo.",
                "no_last_deleted": "⚠️ No last deleted shape to redo.",
                "cannot_activate": "⚠️ Redo unavailable: couldn't activate the session's Paint window.",
            }
            say(msg_map.get(reason, "⚠️ Redo failed."))

    elif action == "delete":
        # If explicitly "previous" mentioned OR both shape & position & size absent → delete previous
        if previous_flag or (not shape and not ents.get("position") and not size_filter):
            res = Control.delete_previous(st.session_state.session_name)
            if res.get("ok"):
                deleted = res.get("deleted", {})
                stype = deleted.get("type", "shape")
                spos = deleted.get("position")
                ssize = deleted.get("size")
                if spos and ssize:
                    say(f"🗑️ Deleted **{ssize} {stype}** at **{spos}**.")
                elif spos:
                    say(f"🗑️ Deleted **{stype}** at **{spos}**.")
                else:
                    say(f"🗑️ Deleted previous **{stype}**.")
            else:
                say("⚠️ No shapes to delete." if res.get("reason") == "no_shapes" else "⚠️ Delete failed.")
        else:
            # Allow delete by any combination of shape / position / size
            res = Control.delete_by(
                st.session_state.session_name,
                shape or "",
                position=resolved_pos if ents.get("position") else None,
                size=size_filter
            )
            if res.get("ok"):
                deleted = res.get("deleted", {})
                dpos  = deleted.get("position")
                dsize = deleted.get("size")
                stype = shape or deleted.get("type", "shape")
                parts = ["**" + (f"{dsize} " if dsize else "") + f"{stype}**"]
                if dpos:
                    parts.append(f"at **{dpos}**")
                say("🗑️ Deleted " + " ".join(parts) + ".")
            else:
                if res.get("reason") == "no_match":
                    want_shape = shape or "shape"
                    want_pos   = (resolved_pos if ents.get("position") else None)
                    want_size  = size_filter
                    bits = []
                    bits.append(f"**{want_shape}**")
                    if want_size:
                        bits.append(f"size **{want_size}**")
                    if want_pos:
                        bits.append(f"at **{want_pos}**")
                    say("⚠️ Couldn’t find " + " ".join(bits) + " to delete.")
                else:
                    say("⚠️ Delete failed.")

    elif action == "draw":
        if not shape:
            say("⚠️ I didn’t detect a shape to draw.")
        else:
            res = Control.ms_paint(
                st.session_state.session_name,
                action="draw",
                shape=shape,
                position=resolved_pos,
                size=resolved_size
            )
            if not isinstance(res, dict):
                say("⚠️ Something went wrong while drawing.")
            elif not res.get("ok", False):
                reason = res.get("reason")
                if reason == "occupied":
                    say(f"⚠️ That area (**{resolved_pos}**) already has a drawing. Delete it first.")
                elif reason == "no_shape":
                    say("⚠️ No shape specified.")
                elif reason == "unknown_shape":
                    say(f"⚠️ Unknown shape `{res.get('shape')}`.")
                    Control._minimize_paint_window()
                else:
                    say("⚠️ Draw failed.")
            else:
                say(f"🖌️ Drawing **{shape}** at **{resolved_pos}** (size **{resolved_size}**).")

    else:
        say("🤔 Sorry, I couldn’t understand that. Try: “draw a big house at top right”, “delete sun at top”, or “redo”.")

    # One-time refresh remains as a safety (should rarely trigger now)
    if is_first_message_of_blank_session and not st.session_state.did_first_write_rerun_done:
        st.session_state.did_first_write_rerun_done = True
        st.rerun()

# -------------------- Sidebar (reads updated CSV in the SAME run) --------------------
with st.sidebar:
    st.header("Chats")

    if st.button("🆕 New chat", use_container_width=True):
        if st.session_state.session_name:
            _save_or_update_chat(
                st.session_state.session_name,
                st.session_state.chat,
                st.session_state.title,
                st.session_state.created_at or datetime.now().strftime(DATE_FMT)
            )

        st.session_state.session_name = ""
        st.session_state.chat = []
        st.session_state.title = ""
        st.session_state.created_at = ""
        st.session_state.did_first_write = False
        st.session_state.did_first_write_rerun_done = False
        st.rerun()

    st.markdown("---")
    chats = _load_all_chats()
    if not chats:
        st.caption("No chats yet. Start one!")
    else:
        for c in chats:
            is_active = (c["id"] == st.session_state.session_name)
            row = st.columns([1, 0.17])

            title_txt = _truncate(c['title'] or '(untitled)')
            btn_label_md = f"**{c['id']}** — {title_txt}"

            with row[0]:
                if is_active:
                    safe_id = html.escape(c["id"])
                    safe_title = html.escape(title_txt)
                    st.markdown(
                        f"""
                        <div style="
                            padding: 8px 10px;
                            border-radius: 8px;
                            background: #4b4b4c;
                            border: 1px solid #efeff1;
                            margin-bottom: 4px;
                            line-height: 1.25;
                            font-size: 1.0rem;
                            color: #ffffff;
                            text-align: center;
                        ">
                        <span style="font-weight: 600;">{safe_id}</span> — {safe_title}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    if st.button(btn_label_md, key=f"btn-{c['id']}", use_container_width=True):
                        st.session_state.session_name = c["id"]
                        st.session_state.chat = c["messages"]
                        st.session_state.title = c["title"]
                        st.session_state.created_at = c["created_at"] or datetime.now().strftime(DATE_FMT)
                        st.session_state.did_first_write = True
                        st.session_state.did_first_write_rerun_done = False
                        st.rerun()

            with row[1]:
                # Popover: ONLY Delete (rename removed entirely)
                with st.popover("⋮", use_container_width=True):
                    st.caption(f"Manage **{c['id']}**")
                    if st.button("Delete", key=f"do_delete_{c['id']}"):
                        _delete_chat_everywhere(c["id"])
                        st.success("Deleted.")
                        if st.session_state.session_name == c["id"]:
                            st.session_state.session_name = ""
                            st.session_state.chat = []
                            st.session_state.title = ""
                            st.session_state.created_at = ""
                        st.rerun()

# -------------------- Render transcript (single source of truth) --------------------
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
