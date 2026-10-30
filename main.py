# main.py
import json
import sys
import numpy as np
import pandas as pd
from random import choice, random, seed

# ---------------------------------------------------------
# Чтобы print с unicode-символами не падал в cp1251
# ---------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# =========================================================
# 0. КОНСТАНТЫ
# =========================================================
STATS = ['Скорость', 'Удар', 'Пас', 'Отбор', 'Верх', 'Тактика']

POS_TO_GROUP = {'З': 'DEF', 'П': 'MID', 'Н': 'ATT', 'В': 'GK', 'Т': 'COACH'}
GROUP = {
    'ЛЗ': 'DEF', 'ПЗ': 'DEF', 'ЦЗ': 'DEF', 'ОПЗ': 'DEF',
    'ЛПЗ': 'MID', 'ППЗ': 'MID', 'ЦПЗ': 'MID', 'АПЗ': 'MID',
    'ЦН': 'ATT', 'ОФ': 'ATT',
    'В': 'GK', 'Т': 'COACH',
}

PERK_UNIVERSAL_MID = 22
PERK_PRESS         = 26
PERK_WING          = 23

READINESS_TAU = 20 / np.log(200)
READINESS_MAX = 20

PHANTOM_NAME = 'Н.И.'

# =========================================================
# 1. ЗАГРУЗКА
# =========================================================
path_df = 'players.xlsx'

def load_data(path=path_df, sheet='Игроки'):
    frame = pd.read_excel(path, sheet_name=sheet)

    def parse_club(s):
        if pd.isna(s):
            return pd.Series([None, None])
        s = str(s).strip()
        if s.endswith(')') and '(' in s:
            name, code = s.rsplit('(', 1)
            return pd.Series([name.strip(), code[:-1].strip()])
        return pd.Series([s, None])

    frame[['клуб_название', 'клуб_страна']] = frame['клуб'].apply(parse_club)
    frame['Имя_чистое'] = frame['Имя'].str.replace(r'\s*\(В\)\s*$', '', regex=True)
    frame['Перки'] = frame['Перки'].fillna(0).astype(int)
    if 'Состояние' in frame.columns:
        frame['Состояние'] = frame['Состояние'].fillna(0).astype(int)
    else:
        frame['Состояние'] = 0
    return frame

# =========================================================
# 2. ГОТОВНОСТЬ
# =========================================================
def _readiness_delta():
    sign = choice([-1, 1])
    ks = np.arange(0, READINESS_MAX + 1)
    w = np.exp(-ks / READINESS_TAU)
    w /= w.sum()
    return sign * int(np.random.choice(ks, p=w))

def roll_readiness(row):
    delta = _readiness_delta()
    state = int(row.get('Состояние', 0) or 0)
    return int(np.clip(delta + state, -READINESS_MAX, READINESS_MAX))

def recalc_readiness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['Готовность'] = frame.apply(roll_readiness, axis=1)
    return frame

# =========================================================
# 3. ШТРАФЫ И ЭФФЕКТИВНЫЙ РЕЙТИНГ
# =========================================================
def _parse_perks(v):
    if pd.isna(v) or v == 0:
        return []
    return [int(x) for x in str(v).split(',') if x.strip().isdigit()]

def position_penalty(row, target_pos):
    perks = _parse_perks(row.get('Перки'))

    for i, col in enumerate(['спец1', 'спец2', 'спец3']):
        if row.get(col) == target_pos:
            return [0.0, 0.02, 0.05][i]

    base_group = POS_TO_GROUP.get(row.get('позиция'))
    tg = GROUP.get(target_pos)

    if PERK_UNIVERSAL_MID in perks and base_group == 'MID' and tg == 'MID':
        return 0.0
    if PERK_PRESS in perks and target_pos in ('ЦЗ', 'ОПЗ'):
        return 0.0
    if PERK_WING in perks and target_pos in ('ЛЗ', 'ПЗ', 'ЛПЗ', 'ППЗ'):
        return 0.0

    if tg == base_group:
        return 0.10
    return 0.90

def effective_rating(row, target_pos):
    penalty = position_penalty(row, target_pos)
    ready = int(row.get('Готовность', 0)) / 100.0
    base = row[STATS].astype(float).mean()
    return base * (1 - penalty) * (1 + ready)

# =========================================================
# 4. СХЕМЫ
# =========================================================
SCHEMAS_PATH = 'schemas.json'

def load_schemas(path=SCHEMAS_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        schemas = json.load(f)

    filtered = {}
    for name, vec in schemas.items():
        cz, fz, opz, cpz, fp, apz, of, cn = vec
        if (of + cn) >= 1 and (cz + fz) <= 5 and sum(vec) == 10:
            filtered[name] = vec
        else:
            print(f'[warn] Схема {name} отфильтрована: '
                  f'нападающих={of + cn}, защитников={cz + fz}, сумма={sum(vec)}')
    return filtered

def schema_to_positions(vec):
    """Ровно 10 полевых позиций (без вратаря)."""
    cz, fz, opz, cpz, fp, apz, of, cn = vec
    out = []
    out += ['ЦЗ'] * cz
    out += ['ЛЗ', 'ПЗ'] * (fz // 2) + (['ЛЗ'] if fz % 2 else [])
    out += ['ОПЗ'] * opz + ['ЦПЗ'] * cpz
    out += ['ЛПЗ', 'ППЗ'] * (fp // 2) + (['ЛПЗ'] if fp % 2 else [])
    out += ['АПЗ'] * apz + ['ОФ'] * of + ['ЦН'] * cn

    if len(out) > 10:
        out = out[:10]
    while len(out) < 10:
        out.append('ЦПЗ')
    return out

# =========================================================
# 5. ДОБИВКА ФАНТОМАМИ
# =========================================================
def ensure_goalkeeper(team: pd.DataFrame, penalty: float = 30.0) -> pd.DataFrame:
    """
    Если в команде нет игрока с позиция == 'В',
    добавляем фантомного вратаря 'Н.И. (В)' с рейтингом
    = mean(ОР полевых) - penalty.
    """
    if len(team) == 0:
        return team
    if (team['позиция'] == 'В').any():
        return team

    field_players = team[team['позиция'] != 'Т']
    if len(field_players) == 0:
        return team

    avg_or = float(field_players['ОР'].mean())
    phantom_rating = max(0.0, avg_or - penalty)

    template = field_players.iloc[0].copy()
    r = template.copy()
    r['Имя'] = f'{PHANTOM_NAME} (В)'
    r['Имя_чистое'] = f'{PHANTOM_NAME} (В)'
    r['позиция'] = 'В'
    r['спец1'] = 'В'
    r['спец2'] = None
    r['спец3'] = None
    for s in STATS:
        r[s] = phantom_rating
    r['ОР'] = phantom_rating
    r['Готовность'] = 0
    r['Перки'] = 0
    r['Состояние'] = 0

    phantom = pd.DataFrame([r])
    base_idx = (team.index.max() + 1) if len(team) else 0
    phantom.index = [base_idx]

    return pd.concat([team, phantom], axis=0)

def augment_with_phantoms(team: pd.DataFrame,
                          max_total: int = 22,
                          penalty: float = 30.0) -> pd.DataFrame:
    """
    Добивает команду фантомами до max_total игроков:
      - сначала гарантируем вратаря (через ensure_goalkeeper);
      - затем добиваем полевыми 'Н.И. N' с рейтингом mean(ОР полевых) - penalty.
    """
    if len(team) == 0:
        return team

    # 1) вратарь обязательно
    team = ensure_goalkeeper(team, penalty=penalty)

    # 2) средний ОР полевых (без вратаря и тренера)
    field_players = team[(team['позиция'] != 'Т') & (team['позиция'] != 'В')]
    if len(field_players) == 0:
        # если совсем никого — берём вратаря как эталон
        field_players = team[team['позиция'] != 'Т']
    avg_or = float(field_players['ОР'].mean())
    phantom_rating = max(0.0, avg_or - penalty)

    needed = max(0, max_total - len(team))
    if needed == 0:
        return team

    template = field_players.iloc[0].copy()

    rows = []
    for k in range(needed):
        r = template.copy()
        r['Имя'] = f'{PHANTOM_NAME} {k + 1}'
        r['Имя_чистое'] = f'{PHANTOM_NAME} {k + 1}'
        r['позиция'] = 'П'
        r['спец1'] = 'ЦПЗ'
        r['спец2'] = None
        r['спец3'] = None
        for s in STATS:
            r[s] = phantom_rating
        r['ОР'] = phantom_rating
        r['Готовность'] = 0
        r['Перки'] = 0
        r['Состояние'] = 0
        r['сборная'] = template.get('сборная')
        r['клуб'] = template.get('клуб')
        r['клуб_название'] = template.get('клуб_название')
        r['клуб_страна'] = template.get('клуб_страна')
        rows.append(r)

    phantom_df = pd.DataFrame(rows)
    base_idx = (team.index.max() + 1) if len(team) else 0
    phantom_df.index = range(base_idx, base_idx + len(phantom_df))

    return pd.concat([team, phantom_df], axis=0)

# =========================================================
# 6. ПОДБОР СОСТАВА
# =========================================================
def pick_lineup(players: pd.DataFrame, schema_vec, seed_val=None):
    if seed_val is not None:
        np.random.seed(seed_val)

    positions = schema_to_positions(schema_vec)
    N = len(players)

    lineup = {}
    used = set()

    gk = players[players['позиция'] == 'В']
    if len(gk):
        gk_best = gk.sort_values('ОР', ascending=False).iloc[0]
        lineup['В'] = [gk_best]
        used.add(gk_best.name)

    R = np.zeros((N, len(positions)))
    for i, (_, p) in enumerate(players.iterrows()):
        for j, pos in enumerate(positions):
            R[i, j] = effective_rating(p, pos)

    pos_to_orig = {i: players.index[i] for i in range(N)}
    orig_to_pos = {v: k for k, v in pos_to_orig.items()}
    used_pos_idx = {orig_to_pos[idx] for idx in used if idx in orig_to_pos}

    # фантомы (Н.И.) должны выбираться только если реальных игроков не хватает.
    # Даём им штраф в матрице рейтингов.
    is_phantom = np.array([
        str(players.iloc[i].get('Имя_чистое', '')).startswith('Н.И.')
        for i in range(N)
    ])
    # штраф 30% к рейтингу — реальный игрок со средним ОР 60 всё равно
    # выигрывает у фантома с 43.6
    R = R * np.where(is_phantom[:, None], 0.7, 1.0)

    def cnt(j):
        # количество реальных игроков, способных закрыть позицию
        real_mask = ~is_phantom
        return int((R[real_mask, j] > 0).sum())

    order = sorted(range(len(positions)), key=cnt)

    chosen = {}
    for j in order:
        avail = [i for i in range(N)
                if i not in chosen.values()
                and i not in used_pos_idx
                and players.iloc[i]['позиция'] not in ('В', 'Т')]
        if not avail:
            break
        scores = R[avail, j]
        mx = scores.max()
        tops = [i for i, s in zip(avail, scores) if abs(s - mx) < 1e-9]
        chosen[j] = int(np.random.choice(tops))

    improved = True
    while improved:
        improved = False
        keys = list(chosen.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                j1, j2 = keys[a], keys[b]
                i1, i2 = chosen[j1], chosen[j2]
                if R[i1, j1] + R[i2, j2] < R[i1, j2] + R[i2, j1] - 1e-9:
                    chosen[j1], chosen[j2] = i2, i1
                    improved = True

    if len(chosen) > 10:
        sorted_j = sorted(chosen.keys(), key=lambda j: -R[chosen[j], j])[:10]
        chosen = {j: chosen[j] for j in sorted_j}

    for j, i in chosen.items():
        orig_idx = pos_to_orig[i]
        pos = positions[j]
        player = players.loc[orig_idx]
        lineup.setdefault(pos, []).append(player)
        used.add(orig_idx)

    total = 0.0
    for pos, plist in lineup.items():
        for pl in plist:
            if pos == 'В':
                total += float(pl['ОР'])
            else:
                total += effective_rating(pl, pos)

    return lineup, total, used

# =========================================================
# 7. СКАМЕЙКА
# =========================================================
def pick_bench(players: pd.DataFrame, schema_vec, used_idx, max_bench=10):
    """
    Скамейка: только реальные игроки. Фантомы (Н.И.) — только если
    реальных игроков не хватает, и не больше 3 штук, чтобы не засорять.
    """
    positions = schema_to_positions(schema_vec)

    rest = players.drop(index=list(used_idx), errors='ignore').copy()
    rest = rest[rest['позиция'] != 'Т']
    rest_no_gk = rest[rest['позиция'] != 'В'].copy()

    # отделяем реальных и фантомов
    real_mask = ~rest_no_gk['Имя_чистое'].astype(str).str.startswith('Н.И.')
    real_players = rest_no_gk[real_mask].copy()
    phantoms = rest_no_gk[~real_mask].copy()

    bench = []
    bench_idx = set()

    # 1) по одному лучшему на каждую уникальную позицию схемы — только из реальных
    for pos in sorted(set(positions)):
        cand = real_players[~real_players.index.isin(bench_idx)]
        if len(cand) == 0:
            break
        cand = cand.copy()
        cand['score'] = cand.apply(lambda r: effective_rating(r, pos), axis=1)
        best = cand.sort_values('score', ascending=False).iloc[0]
        bench.append((pos, best))
        bench_idx.add(best.name)
        if len(bench) >= max_bench:
            break

    # 2) добираем лучших реальных по ОР
    rest2 = real_players[~real_players.index.isin(bench_idx)].copy()
    rest2 = rest2.sort_values('ОР', ascending=False)
    for _, p in rest2.iterrows():
        if len(bench) >= max_bench:
            break
        real_pos = p.get('спец1') or p.get('позиция') or '—'
        bench.append((str(real_pos), p))

    # 3) только если реальных совсем мало — добавляем максимум 3 фантома
    if len(bench) < 5 and len(phantoms) > 0:
        ph_sorted = phantoms.sort_values('ОР', ascending=False).head(3)
        for _, p in ph_sorted.iterrows():
            if len(bench) >= max_bench:
                break
            real_pos = p.get('спец1') or p.get('позиция') or '—'
            bench.append((str(real_pos), p))

    # тренер — отдельно
    coach_df = players[players['позиция'] == 'Т']
    coach = coach_df.sort_values('ОР', ascending=False).iloc[0] if len(coach_df) else None

    return bench, coach

# =========================================================
# 8. АВТО-ПОДБОР СХЕМЫ
# =========================================================
def pick_best_schema(players: pd.DataFrame, schemas: dict):
    results = {}
    for name, vec in schemas.items():
        _, total, _ = pick_lineup(players, vec)
        results[name] = total
    if not results:
        return None, {}
    best_val = max(results.values())
    tops = [k for k, v in results.items() if abs(v - best_val) < 1e-9]
    return choice(tops), results

# =========================================================
# 9. УТИЛИТЫ
# =========================================================
def get_team(frame: pd.DataFrame, kind: str, name: str) -> pd.DataFrame:
    if kind == 'сборная':
        return frame[frame['сборная'] == name].copy()
    if kind == 'клуб':
        return frame[frame['клуб_название'] == name].copy()
    raise ValueError(kind)

def perk_display(perk_value):
    try:
        p = int(perk_value)
    except Exception:
        return ''
    if p == 0:
        return ''
    return f'★{p}'

# =========================================================
# ТРЕНЕРСКОЕ СООТВЕТСТВИЕ
# =========================================================

F_MAX = 0.25   # потолок усиления недоминирующих статов (при тренере 100)

def _dominant_idx(player):
    vals = [float(player[s]) for s in STATS]
    return int(np.argmax(vals))

def coach_profile(coach):
    """dict s -> f(c[s]) ∈ [0, F_MAX], или None."""
    if coach is None:
        return None
    return {s: F_MAX * (float(coach[s]) / 100.0) for s in STATS}

def apply_coach(player, prof):
    """Доминанта не меняется; остальные статы × (1 + f(c[dom]))."""
    base = {s: float(player[s]) for s in STATS}
    if prof is None:
        return base
    dom_i = _dominant_idx(player)
    boost = prof[STATS[dom_i]]
    return {s: (base[s] if i == dom_i else base[s] * (1.0 + boost))
            for i, s in enumerate(STATS)}

def apply_coach_to_team(team: pd.DataFrame, coach) -> pd.DataFrame:
    prof = coach_profile(coach)
    if prof is None:
        return team.copy()

    out = team.copy()

    # 1) база (эти колонки создаются как float — ок)
    for s in STATS:
        out[f'{s}_база'] = out[s].astype(float)

    # 2) ВАЖНО: приводим сами STATS к float, иначе запись упадёт
    for s in STATS:
        out[s] = out[s].astype(float)

    # 3) ОР тоже float — на случай, если будешь писать в него
    out['ОР'] = out['ОР'].astype(float)

    field_mask = out['позиция'] != 'Т'
    field = out.loc[field_mask, STATS]          # уже float, astype не нужен

    dom_idx = field.values.argmax(axis=1)
    boost = np.array([prof[STATS[i]] for i in dom_idx])

    eff = field.values.copy()
    for i in range(len(STATS)):
        m = dom_idx != i
        eff[m, i] = field.values[m, i] * (1.0 + boost[m])

    # 4) теперь запись безопасна — и колонки, и eff — float
    out.loc[field_mask, STATS] = eff
    out['Тренер_буст'] = 0.0
    out.loc[field_mask, 'Тренер_буст'] = boost

    # 5) ОР_эфф отдельной колонкой, базовый ОР не трогаем
    out['ОР_эфф'] = out['ОР'].astype(float)
    out.loc[field_mask, 'ОР_эфф'] = eff.mean(axis=1)

    return out