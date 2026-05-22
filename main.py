import io
import itertools
import json
import os
import random
import secrets
from collections import defaultdict
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Basic認証（環境変数 ADMIN_USER / ADMIN_PASS で設定）
_basic = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(_basic)):
    ok_user = secrets.compare_digest(credentials.username, os.getenv("ADMIN_USER", "admin"))
    ok_pass = secrets.compare_digest(credentials.password, os.getenv("ADMIN_PASS", "changeme"))
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="認証が必要です",
            headers={"WWW-Authenticate": "Basic"},
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE_FILE = Path("state.json")

DEFAULT_STATE = {
    "mode": "disciple",
    "match_count": 0,
    "participants": [],
    "current_match": None,
    "match_history": []
}


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return DEFAULT_STATE.copy()


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class Participant(BaseModel):
    name: str
    role: str  # "master" or "disciple"


class MatchConfirm(BaseModel):
    winner_team: int  # 1 or 2
    team1: list[str]
    team2: list[str]
    match_type: str  # "disciple" or "master"


class ModeChange(BaseModel):
    mode: str  # "disciple" or "master"


@app.get("/api/state")
def get_state():
    return load_state()


@app.get("/api/participants")
def get_participants():
    state = load_state()
    return state["participants"]


@app.post("/api/participants")
def add_participant(p: Participant):
    if p.role not in ("master", "disciple"):
        raise HTTPException(400, "role must be 'master' or 'disciple'")
    state = load_state()
    if any(x["name"] == p.name for x in state["participants"]):
        raise HTTPException(400, f"{p.name} は既に登録済みです")
    state["participants"].append({"name": p.name, "role": p.role, "appearances": 0})
    save_state(state)
    return {"ok": True}


@app.delete("/api/participants/{name}")
def delete_participant(name: str):
    state = load_state()
    before = len(state["participants"])
    state["participants"] = [x for x in state["participants"] if x["name"] != name]
    if len(state["participants"]) == before:
        raise HTTPException(404, f"{name} が見つかりません")
    save_state(state)
    return {"ok": True}


def _build_disciple_history(match_history: list) -> tuple[dict, set]:
    """履歴から (ペア出現回数dict, 対戦済みマッチアップset) を生成"""
    pair_counts: dict[frozenset, int] = defaultdict(int)
    matchup_set: set[frozenset] = set()
    for match in match_history:
        if match.get("type") != "disciple":
            continue
        t1, t2 = match.get("team1", []), match.get("team2", [])
        for team in (t1, t2):
            for a, b in itertools.combinations(team, 2):
                pair_counts[frozenset([a, b])] += 1
        matchup_set.add(frozenset([frozenset(t1), frozenset(t2)]))
    return pair_counts, matchup_set


def _score_split(t1: list, t2: list, pair_counts: dict, matchup_set: set) -> tuple:
    """スコア: (未経験ペア数, マッチアップが初めてかどうか)。高いほど良い。"""
    novel_pairs = sum(
        1 for team in (t1, t2)
        for a, b in itertools.combinations(team, 2)
        if pair_counts[frozenset([a, b])] == 0
    )
    matchup_novel = 1 if frozenset([frozenset(t1), frozenset(t2)]) not in matchup_set else 0
    return (novel_pairs, matchup_novel)


def _all_splits(names: list[str]) -> list[tuple[list, list]]:
    """4人の全3通りの2v2分け方を返す"""
    a, b, c, d = names
    return [
        ([a, b], [c, d]),
        ([a, c], [b, d]),
        ([a, d], [b, c]),
    ]


def _pick_next_disciple(state: dict) -> dict | None:
    disciples = [p for p in state["participants"] if p["role"] == "disciple"]
    if len(disciples) < 4:
        return None

    pair_counts, matchup_set = _build_disciple_history(state["match_history"])

    # 出場回数の少ない順にソート（同数はランダム）
    sorted_d = sorted(disciples, key=lambda x: (x["appearances"], random.random()))

    # 4番目の出場回数以下の全プレイヤーが候補プール
    cutoff = sorted_d[3]["appearances"]
    pool = [p for p in sorted_d if p["appearances"] <= cutoff]

    # 候補の全C(n,4)組み合わせを試す（多すぎる場合は30件サンプル）
    combos = list(itertools.combinations(pool, 4))
    if len(combos) > 30:
        combos = random.sample(combos, 30)

    best_split: tuple[list, list] | None = None
    best_score = (-1, -1)

    for subset in combos:
        names = [p["name"] for p in subset]
        for t1, t2 in _all_splits(names):
            score = _score_split(t1, t2, pair_counts, matchup_set)
            if score > best_score:
                best_score = score
                best_split = (t1, t2)

    if best_split is None:
        names = [p["name"] for p in sorted_d[:4]]
        random.shuffle(names)
        best_split = (names[:2], names[2:])

    return {"team1": best_split[0], "team2": best_split[1], "type": "disciple"}


def _pick_next_master(state: dict) -> dict | None:
    masters = [p for p in state["participants"] if p["role"] == "master"]
    if len(masters) < 2:
        return None

    # 5人以上の場合はランダムに4人を抽選して必ず2v2にする
    selected = random.sample(masters, 4) if len(masters) >= 5 else masters
    names = [p["name"] for p in selected]
    random.shuffle(names)
    return {"team1": names[:2], "team2": names[2:], "type": "master"}


@app.post("/api/match/next-disciple")
def next_disciple():
    state = load_state()
    match = _pick_next_disciple(state)
    if match is None:
        raise HTTPException(400, "弟子が4人以上必要です")
    state["current_match"] = match
    save_state(state)
    return match


@app.post("/api/match/next-master")
def next_master():
    state = load_state()
    match = _pick_next_master(state)
    if match is None:
        raise HTTPException(400, "師匠が2人以上必要です")
    state["current_match"] = match
    save_state(state)
    return match


@app.post("/api/match/confirm")
def confirm_match(body: MatchConfirm):
    state = load_state()
    state["match_count"] += 1
    all_players = body.team1 + body.team2
    for p in state["participants"]:
        if p["name"] in all_players:
            p["appearances"] += 1
    history_entry = {
        "match_no": state["match_count"],
        "type": body.match_type,
        "team1": body.team1,
        "team2": body.team2,
        "winner": body.winner_team,
    }
    state["match_history"].append(history_entry)
    state["current_match"] = None
    save_state(state)
    return {"ok": True, "match_no": state["match_count"]}


@app.post("/api/mode")
def set_mode(body: ModeChange):
    if body.mode not in ("disciple", "master"):
        raise HTTPException(400, "mode must be 'disciple' or 'master'")
    state = load_state()
    state["mode"] = body.mode
    save_state(state)
    return {"ok": True, "mode": body.mode}


@app.post("/api/reset")
def reset():
    save_state(DEFAULT_STATE.copy())
    return {"ok": True}


@app.post("/api/import-excel")
async def import_excel(file: UploadFile = File(...)):
    try:
        import pandas as pd
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content), header=0)
    except Exception as e:
        raise HTTPException(400, f"Excelファイルの読み込みに失敗しました: {e}")

    # 列名を正規化（.1サフィックス付きは師匠の出欠列）
    cols = list(df.columns)
    # 期待列: キャラ, 弟子(敬称略), 出欠, 師匠(敬称略), 出欠.1
    if len(cols) < 5:
        raise HTTPException(400, f"列数が不足しています。列: {cols}")

    disciple_name_col = cols[1]
    disciple_attend_col = cols[2]
    master_name_col = cols[3]
    master_attend_col = cols[4]

    state = load_state()
    existing = {p["name"] for p in state["participants"]}
    added = []
    skipped = []

    ATTEND_OK = {"〇", "☆"}

    for _, row in df.iterrows():
        d_name = row[disciple_name_col]
        d_attend = row[disciple_attend_col]
        if pd.notna(d_name) and pd.notna(d_attend) and str(d_attend).strip() in ATTEND_OK:
            name = str(d_name).strip()
            if name and name not in existing:
                state["participants"].append({"name": name, "role": "disciple", "appearances": 0})
                existing.add(name)
                added.append({"name": name, "role": "disciple"})
            elif name in existing:
                skipped.append(name)

        m_name = row[master_name_col]
        m_attend = row[master_attend_col]
        if pd.notna(m_name) and pd.notna(m_attend) and str(m_attend).strip() in ATTEND_OK:
            name = str(m_name).strip()
            if name and name not in existing:
                state["participants"].append({"name": name, "role": "master", "appearances": 0})
                existing.add(name)
                added.append({"name": name, "role": "master"})
            elif name in existing:
                skipped.append(name)

    save_state(state)
    return {"ok": True, "added": added, "skipped": skipped}


@app.get("/admin", response_class=HTMLResponse)
def admin(_: None = Depends(verify_admin)):
    return Path("static/admin.html").read_text(encoding="utf-8")


@app.get("/obs", response_class=HTMLResponse)
def obs():
    return Path("static/obs.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def root():
    return '<meta http-equiv="refresh" content="0; url=/obs">'
