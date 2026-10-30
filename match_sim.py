# match_sim.py
"""
Симуляция футбольного матча.

Ядро — пуассоновская модель: итоговый счёт разыгрывается один раз в начале
матча как Пуассон(xg_home) и Пуассон(xg_away). Дальше эпизоды лишь
«реализуют» эти голы и дают статистику (удары, углы, карточки, замены,
травмы, усталость).

Все ключевые коэффициенты — в блоке CONFIG в начале файла.
"""
import numpy as np
from random import random, choice, randint, shuffle

# =========================================================
# CONFIG — ВСЕ КОЭФФИЦИЕНТЫ В ОДНОМ МЕСТЕ
# =========================================================
CONFIG = {
    # ---------- ГОТОВНОСТЬ ----------
    'readiness_tau': 20 / np.log(200),

    # ---------- ОЖИДАЕМЫЕ ГОЛЫ (Пуассон) ----------
    'xg_slope': 8.00,
    'xg_scale': 60,
    'xg_max': 12.00,
    'home_adv_or_bonus': 8.0,
    'home_strength_bonus': 0.06,

    # ---------- ВЛАДЕНИЕ ----------
    'possession_scale': 15,
    'possession_clip_min': 0.15,
    'possession_clip_max': 0.85,

    # ---------- СУПЕРМОТИВАЦИЯ ----------
    'super_motivation_chance': 0.10,
    'super_motivation_mult': 1.25,

    # ---------- ЛЕГЕНДАРНЫЙ МАТЧ ----------
    # Редкое событие для ОДНОЙ случайной команды: она играет на пределе
    # возможностей (+50% к боевым статам). Команда выбирается случайно,
    # не зависит от силы.
    'legendary_match_chance': 0.05,   # ~1.5% (1 из ~67 матчей)
    'legendary_match_mult': 2.00,

    # ---------- НИЧЬИ ----------
    'draw_threshold': 20,
    'draw_max_chance': 0.35,
    'draw_min_chance': 0.01,
    'draw_score_gap_max': 2,

    # ---------- УСТАЛОСТЬ ----------
    'fatigue_threshold': 45,
    'fatigue_decay_perks': 70,
    'injury_chance': 0.035,
    'max_subs': 5,

    # ---------- ПРОЧЕЕ ----------
    'assist_chance': 0.75,
    'corner_goal_base': 0.06,
    'corner_goal_slope': 0.08,
    'penalty_goal_base': 0.78,
    'penalty_from_foul': 0.03,
    'fk_goal_base': 0.05,
    'fk_goal_slope': 0.06,
    'yellow_from_foul': 0.10,
    'red_from_foul': 0.05,

    # ---------- НАСТРОЙКИ ПО УМОЛЧАНИЮ ----------
    'default_n_episodes_per_block': 15,
    'default_home_adv_on': True,
    'default_home_adv_strength': 1.0,
}

# =========================================================
# ПЕРКИ → МОДИФИКАТОРЫ
# =========================================================
PERK_MODS = {
    0:  {},
    1:  {'shot': 0.10, 'goal': 0.03, 'range': 0.25},
    2:  {'shot': 0.08, 'goal': 0.02},
    3:  {'dribble': 0.15},
    4:  {'dribble': 0.12, 'shot': 0.05},
    5:  {'shot': 0.10, 'goal': 0.05, 'dribble': 0.05},
    6:  {'shot': 0.05, 'card_receive': -0.05},
    7:  {'goal': 0.10, 'shot': 0.05},
    8:  {'assist': 0.15, 'pass_': 0.05},
    9:  {'shot': 0.07, 'goal': 0.03},
    10: {'pass_': 0.10, 'shot': 0.08, 'dribble': 0.08,
         'tackle': 0.05, 'intercept': 0.05, 'aura': 0.05},
    11: {'tackle': 0.10, 'shot': 0.05, 'card': 0.10},
    12: {'variance': 0.20},
    13: {'aura': 0.15, 'pass_': 0.05, 'shot': 0.05},
    14: {'pass_': 0.05, 'shot': 0.05, 'dribble': 0.05, 'tackle': 0.05},
    15: {'pass_': 0.10, 'dribble': 0.05},
    16: {},
    17: {'assist': 0.10, 'pass_': 0.08},
    18: {'intercept': 0.15, 'tackle': 0.10},
    19: {'dribble': 0.10, 'pass_': 0.08, 'penalty_earn': 0.10},
    20: {'rebound': 0.20},
    21: {'shot': 0.10, 'goal': 0.04, 'range': 0.15},
    22: {'pass_': 0.10},
    23: {'dribble': 0.10, 'speed': 0.10},
    24: {'pass_': 0.20, 'assist': 0.10},
    25: {'variance': 0.30},
    26: {'tackle': 0.10, 'intercept': 0.05},
    27: {'tackle': 0.15},
    28: {'offside_trap': 0.15},
    29: {'save': 0.20},
    30: {'tackle': 0.12},
    31: {'pass_': 0.10, 'clearance': 0.15},
}

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def _parse_perks(v):
    if v is None:
        return []
    try:
        if isinstance(v, float) and np.isnan(v):
            return []
    except Exception:
        pass
    if v == 0:
        return []
    return [int(x) for x in str(v).split(',') if x.strip().isdigit()]

def _perk_sum(perks, key, fatigue=0):
    ft = CONFIG['fatigue_decay_perks']
    if fatigue >= ft:
        mult = max(0.0, 1.0 - (fatigue - ft) / 30.0)
    else:
        mult = 1.0
    total = 0.0
    for p in perks:
        total += PERK_MODS.get(p, {}).get(key, 0.0)
    return total * mult

def _fatigue_mult(fatigue):
    return float(np.clip(1.0 - (fatigue / 100.0) ** 1.3, 0.55, 1.0))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# =========================================================
# СИЛА КОМАНДЫ
# =========================================================
def team_strength(lineup, coach, aura_opp=0.0, fatigue_map=None, home_bonus_mult=1.0):
    """home_bonus_mult — общий множитель боевых статов (дом + мотивация)."""
    if fatigue_map is None:
        fatigue_map = {}

    def collect(pred):
        out = []
        for pos, plist in lineup.items():
            for p in plist:
                if pred(pos):
                    out.append((pos, p))
        return out

    defenders = collect(lambda p: p in ('ЛЗ', 'ПЗ', 'ЦЗ', 'ОПЗ'))
    midfielders = collect(lambda p: p in ('ЛПЗ', 'ППЗ', 'ЦПЗ', 'АПЗ'))
    attackers = collect(lambda p: p in ('ЦН', 'ОФ'))
    gk = collect(lambda p: p == 'В')

    def line_avg(players, stat):
        vals = []
        for pos, p in players:
            f = fatigue_map.get(p['Имя_чистое'], 0)
            perks = _parse_perks(p.get('Перки'))
            v = float(p[stat]) * _fatigue_mult(f)
            v *= (1 + _perk_sum(perks, 'aura', f) * 0.5)
            vals.append(v)
        return float(np.mean(vals)) if vals else 50.0

    all_players = defenders + midfielders + attackers
    aura_team = (float(np.mean([float(p['Аура']) for _, p in all_players]))
                 if all_players else 50.0)

    aura_diff = aura_opp - aura_team
    if   aura_diff <= 5:   aura_mult = 1.00
    elif aura_diff <= 15:  aura_mult = 0.98
    elif aura_diff <= 30:  aura_mult = 0.95
    else:                  aura_mult = 0.90

    coach_stats = {s: 70.0 for s in ['Скорость', 'Удар', 'Пас', 'Отбор', 'Верх', 'Тактика']}
    coach_perks = []
    if coach is not None:
        for s in coach_stats:
            coach_stats[s] = float(coach[s])
        coach_perks = _parse_perks(coach.get('Перки'))

    def perk_sum_players(players, key):
        total = 0.0
        for _, p in players:
            f = fatigue_map.get(p['Имя_чистое'], 0)
            total += _perk_sum(_parse_perks(p.get('Перки')), key, f)
        return total

    hb = home_bonus_mult

    return {
        'pass_mid': line_avg(midfielders, 'Пас') * aura_mult * hb
                    * (1 + perk_sum_players(midfielders, 'pass_'))
                    * (1 + (coach_stats['Пас'] - 70) / 500),
        'tackle_def': line_avg(defenders, 'Отбор') * aura_mult * hb
                      * (1 + perk_sum_players(defenders, 'tackle'))
                      * (1 + (coach_stats['Отбор'] - 70) / 500),
        'tackle_mid': line_avg(midfielders, 'Отбор') * aura_mult * hb
                      * (1 + perk_sum_players(midfielders, 'tackle')),
        'shot_att': line_avg(attackers, 'Удар') * aura_mult * hb
                    * (1 + perk_sum_players(attackers, 'shot'))
                    * (1 + (coach_stats['Удар'] - 70) / 500),
        'shot_mid': line_avg(midfielders, 'Удар') * aura_mult * hb,
        'air_def': line_avg(defenders, 'Верх') * aura_mult * hb,
        'air_att': line_avg(attackers, 'Верх') * aura_mult * hb,
        'gk': (float(gk[0][1]['ОР']) * _fatigue_mult(fatigue_map.get(gk[0][1]['Имя_чистое'], 0))
               if gk else 50.0) * aura_mult * hb,
        'gk_perks': _parse_perks(gk[0][1].get('Перки')) if gk else [],
        'aura': aura_team,
        'aura_mult': aura_mult,
        'coach_perks': coach_perks,
        'coach_stats': coach_stats,
    }

# =========================================================
# УСТАЛОСТЬ
# =========================================================
def _fatigue_growth(player, opp_strength_ratio, minute, position):
    if position == 'В':
        base = randint(1, 3)
    elif position in ('ЦЗ', 'ЛЗ', 'ПЗ'):
        base = randint(8, 14)
    elif position in ('ОПЗ', 'ЦПЗ', 'ЛПЗ', 'ППЗ'):
        base = randint(10, 16)
    elif position in ('АПЗ', 'ОФ', 'ЦН'):
        base = randint(9, 15)
    else:
        base = randint(8, 14)

    if opp_strength_ratio > 1.25:
        base *= 1.20
    elif opp_strength_ratio > 1.10:
        base *= 1.08
    elif opp_strength_ratio < 0.9:
        base *= 0.90

    base *= random() * 0.5 + 0.75

    if minute > 75:
        base *= 1.15
    if minute > 85:
        base *= 1.10

    aura = float(player.get('Аура', 50))
    aura_factor = 1.0 - max(0.0, (aura - 50.0)) / 500.0
    aura_factor = max(0.85, min(1.05, aura_factor))
    base *= aura_factor

    return float(base)

# =========================================================
# СИМУЛЯЦИЯ МАТЧА
# =========================================================
def simulate_match(lineup_home, coach_home,
                   lineup_away, coach_away,
                   bench_home=None, bench_away=None,
                   seed_val=None,
                   n_episodes_per_block=None,
                   home_adv_on=None,
                   home_adv_strength=None,
                   max_substitutions=None):
    if seed_val is not None:
        np.random.seed(seed_val)

    n_episodes_per_block = n_episodes_per_block or CONFIG['default_n_episodes_per_block']
    if home_adv_on is None:
        home_adv_on = CONFIG['default_home_adv_on']
    if home_adv_strength is None:
        home_adv_strength = CONFIG['default_home_adv_strength']
    if max_substitutions is None:
        max_substitutions = CONFIG['max_subs']

    bench_home = bench_home or []
    bench_away = bench_away or []

    def avg_field_or(lineup):
        vals = [float(p['ОР']) for pos, plist in lineup.items() for p in plist]
        return float(np.mean(vals)) if vals else 60.0

    OR_home = avg_field_or(lineup_home)
    OR_away = avg_field_or(lineup_away)

    # ---- домашнее преимущество ----
    home_strength_mult = 1.0
    home_or_bonus = 0.0
    if home_adv_on:
        home_strength_mult = 1.0 + CONFIG['home_strength_bonus'] * home_adv_strength
        home_or_bonus = CONFIG['home_adv_or_bonus'] * home_adv_strength

    # ---- легендарный матч (для ОДНОЙ случайной команды) ----
    legendary_match = False
    legendary_team = None
    if random() < CONFIG['legendary_match_chance']:
        legendary_match = True
        legendary_team = choice(['home', 'away'])

    # ---- мотивация ----
    motivation_home = 1.0
    motivation_away = 1.0
    super_home = False
    super_away = False

    if legendary_match:
        base_mult = CONFIG['legendary_match_mult']
        if legendary_team == 'home':
            motivation_home = base_mult
        else:
            motivation_away = base_mult
    else:
        if random() < CONFIG['super_motivation_chance']:
            motivation_home = CONFIG['super_motivation_mult']
            super_home = True
        if random() < CONFIG['super_motivation_chance']:
            motivation_away = CONFIG['super_motivation_mult']
            super_away = True

    # общий множитель: дом × мотивация
    home_strength_total = home_strength_mult * motivation_home
    away_strength_total = motivation_away

    # ---- усталость (инициализация) ----
    fatigue = {}
    for pos, plist in lineup_home.items():
        for p in plist:
            fatigue[p['Имя_чистое']] = 0.0
    for pos, plist in lineup_away.items():
        for p in plist:
            fatigue[p['Имя_чистое']] = 0.0

    S_home_pre = team_strength(lineup_home, coach_home, fatigue_map=fatigue,
                               home_bonus_mult=home_strength_total)
    S_away_pre = team_strength(lineup_away, coach_away, fatigue_map=fatigue,
                               home_bonus_mult=away_strength_total)

    # ---- предварительная оценка владения ----
    mid_home_avg = (0.40 * S_home_pre['pass_mid'] + 0.60 * OR_home) * home_strength_total
    mid_away_avg = (0.40 * S_away_pre['pass_mid'] + 0.60 * OR_away) * away_strength_total
    diff_avg = mid_home_avg - mid_away_avg
    p_raw_avg = _sigmoid(diff_avg / CONFIG['possession_scale'])
    p_home_poss_avg = 0.50 + 0.55 * (p_raw_avg - 0.5)
    p_home_poss_avg = float(np.clip(p_home_poss_avg,
                                     CONFIG['possession_clip_min'],
                                     CONFIG['possession_clip_max']))

    # ---- xG с учётом доли эпизодов ----
    or_diff_eff = (OR_home * home_strength_total + home_or_bonus) - (OR_away * away_strength_total)
    or_diff_raw = OR_home - OR_away

    # доля xG = логистика от or_diff_eff
    p_home_rate = 1.0 / (1.0 + 10 ** (-or_diff_eff / CONFIG['xg_scale']))
    # доля эпизодов — с учётом xG (то, что уже сделано)
    xg_share_home = p_home_rate
    # итоговый xG = сколько голов команда может забить при своей доле эпизодов
    # исходим из того, что 1 гол ≈ 2.5 удара ≈ 4 эпизода
    total_ep_expected = 6 * n_episodes_per_block   # ~48
    ep_home_expected = total_ep_expected * xg_share_home
    ep_away_expected = total_ep_expected * (1 - xg_share_home)
    # конверсия эпизода в гол ~ 8% для лучших команд, 3% для слабых
    # но мы хотим сохранить пропорциональность через xg_slope
    xg_home = CONFIG['xg_slope'] * xg_share_home * (0.5 + p_home_poss_avg)
    xg_away = CONFIG['xg_slope'] * (1.0 - xg_share_home) * (1.5 - p_home_poss_avg)
    xg_home = min(CONFIG['xg_max'], xg_home)
    xg_away = min(CONFIG['xg_max'], xg_away)

    # ---- целевой счёт по Пуассону (для отображения в expected) ----
    target_home_goals = int(np.random.poisson(xg_home))
    target_away_goals = int(np.random.poisson(xg_away))
    # goals_left — не используется в новой модели, но оставим для совместимости
    goals_left_home = 0
    goals_left_away = 0

    events = []
    score = {'home': 0, 'away': 0}
    stats = {
        'home': {'shots': 0, 'on_target': 0, 'corners': 0, 'fouls': 0,
                 'yellow': 0, 'red': 0, 'possession': 0,
                 'penalties': 0, 'free_kicks': 0, 'subs': 0,
                 'goals_from_pen': 0, 'goals_from_fk': 0,
                 'goals_from_tackle': 0, 'goals_from_corner': 0,
                 'injuries': 0},
        'away': {'shots': 0, 'on_target': 0, 'corners': 0, 'fouls': 0,
                 'yellow': 0, 'red': 0, 'possession': 0,
                 'penalties': 0, 'free_kicks': 0, 'subs': 0,
                 'goals_from_pen': 0, 'goals_from_fk': 0,
                 'goals_from_tackle': 0, 'goals_from_corner': 0,
                 'injuries': 0},
    }
    red_cards = {'home': 0, 'away': 0}
    subs_made = {'home': 0, 'away': 0}
    possession_episodes = {'home': 0, 'away': 0}

    # ===================================================
    # ВЫБОР ИГРОКОВ
    # ===================================================
    def pick_from(lineup, positions):
        cands = []
        for pos in positions:
            for p in lineup.get(pos, []):
                cands.append(p)
        return choice(cands) if cands else None

    def pick_attacker(lineup):
        return pick_from(lineup, ('ЦН', 'ОФ', 'АПЗ', 'ЛПЗ', 'ППЗ', 'ЦПЗ'))
    def pick_defender(lineup):
        return pick_from(lineup, ('ЦЗ', 'ЛЗ', 'ПЗ', 'ОПЗ', 'ЦПЗ'))
    def pick_midfielder(lineup):
        return pick_from(lineup, ('ЦПЗ', 'ОПЗ', 'АПЗ', 'ЛПЗ', 'ППЗ'))
    def pick_gk(lineup):
        gk = lineup.get('В', [])
        return gk[0] if gk else None

    def pick_assist(lineup, exclude):
        cands = []
        for pos, plist in lineup.items():
            for p in plist:
                if p['Имя_чистое'] == exclude:
                    continue
                if pos in ('ЦН', 'ОФ', 'АПЗ', 'ЛПЗ', 'ППЗ', 'ЦПЗ'):
                    cands.append(p)
        if not cands:
            return None
        scored = []
        for p in cands:
            perks = _parse_perks(p.get('Перки'))
            f = fatigue.get(p['Имя_чистое'], 0)
            sc = _perk_sum(perks, 'assist', f) * 10 + random()
            scored.append((sc, p))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    # ===================================================
    # ВЕРОЯТНОСТНЫЕ ФУНКЦИИ
    # ===================================================
    def p_goal_from_shot(att, dfd, gk, home_bonus=0.0):
        a = float(att['Удар']) * _fatigue_mult(fatigue.get(att['Имя_чистое'], 0))
        d = (float(dfd['Отбор']) * _fatigue_mult(fatigue.get(dfd['Имя_чистое'], 0))
             if dfd is not None else 50.0)
        g = float(gk['ОР']) if gk is not None else 50.0
        base = 0.10 + 0.10 * _sigmoid((a - d) / 30) - 0.03 * _sigmoid((g - a) / 35)
        perks = _parse_perks(att.get('Перки'))
        f = fatigue.get(att['Имя_чистое'], 0)
        base *= (1 + _perk_sum(perks, 'goal', f))
        if gk is not None:
            gk_perks = _parse_perks(gk.get('Перки'))
            gk_f = fatigue.get(gk['Имя_чистое'], 0)
            base *= (1 - min(0.30, _perk_sum(gk_perks, 'save', gk_f)))
        base += home_bonus
        return float(np.clip(base, 0.02, 0.60))

    def p_dribble(att, dfd):
        a = float(att['Скорость']) * _fatigue_mult(fatigue.get(att['Имя_чистое'], 0))
        d = (float(dfd['Отбор']) * _fatigue_mult(fatigue.get(dfd['Имя_чистое'], 0))
             if dfd is not None else 50.0)
        perks = _parse_perks(att.get('Перки'))
        f = fatigue.get(att['Имя_чистое'], 0)
        base = 0.45 + 0.25 * _sigmoid((a - d) / 22)
        base *= (1 + _perk_sum(perks, 'dribble', f))
        if dfd is not None:
            dperks = _parse_perks(dfd.get('Перки'))
            df_f = fatigue.get(dfd['Имя_чистое'], 0)
            base *= (1 - min(0.35, _perk_sum(dperks, 'tackle', df_f)))
        return float(np.clip(base, 0.10, 0.82))

    def p_pass(mid, opp_mid):
        a = float(mid['Пас']) * _fatigue_mult(fatigue.get(mid['Имя_чистое'], 0))
        d = (float(opp_mid['Отбор']) * _fatigue_mult(fatigue.get(opp_mid['Имя_чистое'], 0))
             if opp_mid is not None else 50.0)
        perks = _parse_perks(mid.get('Перки'))
        f = fatigue.get(mid['Имя_чистое'], 0)
        base = 0.65 + 0.20 * _sigmoid((a - d) / 25)
        base *= (1 + _perk_sum(perks, 'pass_', f))
        if opp_mid is not None:
            dperks = _parse_perks(opp_mid.get('Перки'))
            df_f = fatigue.get(opp_mid['Имя_чистое'], 0)
            base *= (1 - min(0.30, _perk_sum(dperks, 'intercept', df_f)))
        return float(np.clip(base, 0.28, 0.92))

    def reduce_strength(side):
        if red_cards[side] == 0:
            return 1.0
        return max(0.40, 1.0 - 0.30 * red_cards[side])

    def find_fatigued(lineup, positions):
        cands = []
        for pos in positions:
            for p in lineup.get(pos, []):
                f = fatigue.get(p['Имя_чистое'], 0)
                if f >= CONFIG['fatigue_threshold']:
                    cands.append((pos, p, f))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[2])
        return cands[0]

    def do_replacement(lineup, side, pos_out, player_out, player_in, bench, reason, minute):
        new_list = []
        replaced = False
        for p in lineup.get(pos_out, []):
            if not replaced and p['Имя_чистое'] == player_out['Имя_чистое']:
                new_list.append(player_in)
                replaced = True
            else:
                new_list.append(p)
        lineup[pos_out] = new_list
        fatigue[player_in['Имя_чистое']] = 0.0
        for i, (pos_in, pl_in) in enumerate(bench):
            if pl_in['Имя_чистое'] == player_in['Имя_чистое']:
                bench.pop(i)
                break
        subs_made[side] += 1
        stats[side]['subs'] += 1
        events.append({
            'minute': minute,
            'type': 'substitution' if reason == 'tactical' else 'injury_sub',
            'team': side,
            'player_out': player_out['Имя_чистое'],
            'player_in': player_in['Имя_чистое'],
            'detail': (f"↔ {player_out['Имя_чистое']} → {player_in['Имя_чистое']}"
                       + (' (травма)' if reason == 'injury' else '')),
        })

    def remove_player(lineup, side, pos_out, player_out, minute, reason):
        new_list = [p for p in lineup.get(pos_out, [])
                    if p['Имя_чистое'] != player_out['Имя_чистое']]
        if new_list:
            lineup[pos_out] = new_list
        else:
            lineup.pop(pos_out, None)
        events.append({
            'minute': minute, 'type': 'injury_no_sub', 'team': side,
            'player': player_out['Имя_чистое'],
            'detail': f"{player_out['Имя_чистое']} получил травму, замен нет — в меньшинстве",
        })

    def try_tactical_substitution(lineup, side, minute, bench):
        if subs_made[side] >= max_substitutions:
            return
        if not bench:
            return
        if minute < 55 or random() > 0.55:
            return
        target = find_fatigued(lineup, ('ЦН', 'ОФ', 'АПЗ', 'ЛПЗ', 'ППЗ', 'ЦПЗ', 'ОПЗ'))
        if target is None:
            target = find_fatigued(lineup, ('ЦЗ', 'ЛЗ', 'ПЗ'))
        if target is None:
            return
        pos_out, player_out, fat_out = target
        replacement = None
        for i, (pos_in, player_in) in enumerate(bench):
            if player_in['Имя_чистое'] == player_out['Имя_чистое']:
                continue
            if pos_in == pos_out or player_in.get('спец1') == pos_out:
                replacement = (i, player_in)
                break
        if replacement is None:
            best_i = max(range(len(bench)), key=lambda i: float(bench[i][1]['ОР']))
            replacement = (best_i, bench[best_i][1])
        _, player_in = replacement
        do_replacement(lineup, side, pos_out, player_out, player_in,
                       bench, 'tactical', minute)

    def try_injury(lineup, side, minute, bench):
        if random() > CONFIG['injury_chance']:
            return
        field = []
        for pos, plist in lineup.items():
            for p in plist:
                if pos != 'В':
                    field.append((pos, p))
        if not field:
            return
        weights = [1.0 + fatigue.get(p['Имя_чистое'], 0) / 50.0 for _, p in field]
        total_w = sum(weights)
        r = random() * total_w
        acc = 0.0
        chosen = field[0]
        for (pos, p), w in zip(field, weights):
            acc += w
            if acc >= r:
                chosen = (pos, p)
                break
        pos_out, player_out = chosen
        stats[side]['injuries'] += 1
        if subs_made[side] < max_substitutions and bench:
            replacement = None
            for i, (pos_in, pl_in) in enumerate(bench):
                if pos_in == pos_out or pl_in.get('спец1') == pos_out:
                    replacement = (i, pl_in)
                    break
            if replacement is None:
                best_i = max(range(len(bench)), key=lambda i: float(bench[i][1]['ОР']))
                replacement = (best_i, bench[best_i][1])
            _, player_in = replacement
            do_replacement(lineup, side, pos_out, player_out, player_in,
                           bench, 'injury', minute)
        else:
            remove_player(lineup, side, pos_out, player_out, minute, 'injury')
            red_cards[side] += 1

    # ===================================================
    # МАТЧ
    # ===================================================
    blocks = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]

    for (m_start, m_end) in blocks:
        for side, lineup, opp_or, own_or in (
            ('home', lineup_home, OR_away, OR_home),
            ('away', lineup_away, OR_home, OR_away),
        ):
            opp_ratio = opp_or / max(1.0, own_or)
            for pos, plist in lineup.items():
                for p in plist:
                    name = p['Имя_чистое']
                    fatigue[name] = min(
                        100.0,
                        fatigue.get(name, 0) + _fatigue_growth(p, opp_ratio, m_end, pos)
                    )

        # if m_start >= 46:
        #     sub_minute = randint(m_start, m_end)
        #     try_tactical_substitution(lineup_home, 'home', sub_minute, bench_home)
        #     try_tactical_substitution(lineup_away, 'away', sub_minute, bench_away)
        if m_end >= 55:                      # а не m_start >= 46
            n_attempts = 2 if m_end < 75 else 3
            for _ in range(n_attempts):
                sub_minute = randint(max(55, m_start), m_end)
                try_tactical_substitution(lineup_home, 'home', sub_minute, bench_home)
                try_tactical_substitution(lineup_away, 'away', sub_minute, bench_away)
        injury_minute = randint(m_start, m_end)
        try_injury(lineup_home, 'home', injury_minute, bench_home)
        try_injury(lineup_away, 'away', injury_minute, bench_away)

        S_home = team_strength(lineup_home, coach_home,
                               aura_opp=S_away_pre['aura'], fatigue_map=fatigue,
                               home_bonus_mult=home_strength_total)
        S_away = team_strength(lineup_away, coach_away,
                               aura_opp=S_home_pre['aura'], fatigue_map=fatigue,
                               home_bonus_mult=away_strength_total)

        mid_home = (0.40 * S_home['pass_mid'] + 0.60 * OR_home) * reduce_strength('home') * home_strength_total
        mid_away = (0.40 * S_away['pass_mid'] + 0.60 * OR_away) * reduce_strength('away') * away_strength_total
        diff = mid_home - mid_away
        p_raw = _sigmoid(diff / CONFIG['possession_scale'])
        p_home_poss = 0.50 + 0.55 * (p_raw - 0.5)
        p_home_poss = float(np.clip(p_home_poss,
                                     CONFIG['possession_clip_min'],
                                     CONFIG['possession_clip_max']))

        n_ep = n_episodes_per_block + randint(-2, 2)
        # доля эпизодов = среднее между владением и xG-share
        xg_total = max(0.01, xg_home + xg_away)
        xg_share_home = xg_home / xg_total
        episode_share_home = 0.5 * p_home_poss + 0.5 * xg_share_home
        episode_share_home = float(np.clip(episode_share_home, 0.20, 0.80))
        n_ep_home = int(round(n_ep * episode_share_home))
        n_ep_away = n_ep - n_ep_home
        episode_sides = ['home'] * n_ep_home + ['away'] * n_ep_away
        shuffle(episode_sides)

        for side_ep in episode_sides:
            minute = randint(m_start, m_end)
            home_side = (side_ep == 'home')
            side = 'home' if home_side else 'away'
            lineup_att = lineup_home if home_side else lineup_away
            lineup_def = lineup_away if home_side else lineup_home
            S_att = S_home if home_side else S_away
            S_def = S_away if home_side else S_home
            red_mult_att = reduce_strength(side)

            possession_episodes[side] += 1

            mid = pick_midfielder(lineup_att)
            opp_mid = pick_midfielder(lineup_def)
            if mid is None:
                continue
            pass_ok = random() < p_pass(mid, opp_mid) * red_mult_att

            if not pass_ok:
                if opp_mid is not None and random() < 0.18:
                    events.append({
                        'minute': minute, 'type': 'interception',
                        'team': 'away' if home_side else 'home',
                        'detail': f"{opp_mid['Имя_чистое']} перехватил"
                    })
                continue

            att = pick_attacker(lineup_att)
            if att is None:
                continue
            dfd = pick_defender(lineup_def)
            dribble_ok = random() < p_dribble(att, dfd) * red_mult_att

            if not dribble_ok:
                if dfd is not None and random() < 0.30:
                    stats['away' if home_side else 'home']['fouls'] += 1
                    dfd_perks = _parse_perks(dfd.get('Перки'))
                    df_f = fatigue.get(dfd['Имя_чистое'], 0)
                    card_risk = CONFIG['yellow_from_foul'] + _perk_sum(dfd_perks, 'card', df_f)

                    if random() < card_risk:
                        stats['away' if home_side else 'home']['yellow'] += 1
                        events.append({
                            'minute': minute, 'type': 'yellow',
                            'team': 'away' if home_side else 'home',
                            'player': dfd['Имя_чистое'],
                            'detail': f"ЖК — {dfd['Имя_чистое']}"
                        })
                    if random() < CONFIG['red_from_foul'] + _perk_sum(dfd_perks, 'card', df_f) / 3:
                        stats['away' if home_side else 'home']['red'] += 1
                        red_cards['away' if home_side else 'home'] += 1
                        events.append({
                            'minute': minute, 'type': 'red',
                            'team': 'away' if home_side else 'home',
                            'player': dfd['Имя_чистое'],
                            'detail': f"КК — {dfd['Имя_чистое']}"
                        })

                    if random() < CONFIG['penalty_from_foul']:
                        pen_team = side
                        stats[pen_team]['penalties'] += 1
                        pen_taker = att
                        gk = pick_gk(lineup_def)
                        p_pen = CONFIG['penalty_goal_base']
                        if gk is not None:
                            gk_perks = _parse_perks(gk.get('Перки'))
                            gk_f = fatigue.get(gk['Имя_чистое'], 0)
                            p_pen *= (1 - min(0.30, _perk_sum(gk_perks, 'save', gk_f)))
                        if random() < p_pen and goals_left_home + goals_left_away > 0:
                            if (home_side and goals_left_home > 0) or ((not home_side) and goals_left_away > 0):
                                score[pen_team] += 1
                                if home_side:
                                    goals_left_home -= 1
                                else:
                                    goals_left_away -= 1
                                stats[pen_team]['on_target'] += 1
                                stats[pen_team]['goals_from_pen'] += 1
                                events.append({
                                    'minute': minute, 'type': 'goal',
                                    'team': pen_team,
                                    'scorer': pen_taker['Имя_чистое'],
                                    'assister': None,
                                    'detail': 'с пенальти'
                                })
                                continue
                            else:
                                events.append({
                                    'minute': minute, 'type': 'penalty_miss',
                                    'team': pen_team,
                                    'player': pen_taker['Имя_чистое'],
                                    'detail': f"{pen_taker['Имя_чистое']} не забил пенальти"
                                })
                        else:
                            events.append({
                                'minute': minute, 'type': 'penalty_miss',
                                'team': pen_team,
                                'player': pen_taker['Имя_чистое'],
                                'detail': f"{pen_taker['Имя_чистое']} не забил пенальти"
                            })
                    elif random() < 0.20:
                        fk_team = side
                        stats[fk_team]['free_kicks'] += 1
                        fk_shooter = max(
                            [p for pos, plist in lineup_att.items() for p in plist],
                            key=lambda p: float(p['Удар']) * _fatigue_mult(fatigue.get(p['Имя_чистое'], 0)),
                            default=att
                        )
                        p_fk = CONFIG['fk_goal_base'] + CONFIG['fk_goal_slope'] * _sigmoid(
                            (float(fk_shooter['Удар']) - 70) / 25)
                        if random() < p_fk:
                            if (home_side and goals_left_home > 0) or ((not home_side) and goals_left_away > 0):
                                score[fk_team] += 1
                                if home_side:
                                    goals_left_home -= 1
                                else:
                                    goals_left_away -= 1
                                stats[fk_team]['on_target'] += 1
                                stats[fk_team]['goals_from_fk'] += 1
                                events.append({
                                    'minute': minute, 'type': 'goal',
                                    'team': fk_team,
                                    'scorer': fk_shooter['Имя_чистое'],
                                    'assister': None,
                                    'detail': 'со штрафного'
                                })
                continue

            # -------- УДАР (только считаем, голы разыграем в конце) --------
            stats[side]['shots'] += 1
            gk = pick_gk(lineup_def)
            gk_name = gk['Имя_чистое'] if gk is not None else 'вратарь'

            # Разделяем удары на три типа для статистики и последующей
            # реализации голов. Сама реализация — в конце матча.
            # Точность удара зависит от Удара игрока против ОР вратаря
            att_shot = float(att['Удар']) * _fatigue_mult(fatigue.get(att['Имя_чистое'], 0))
            gk_rating = float(gk['ОР']) if gk is not None else 50.0
            # вероятность попасть в створ ~ от Удара
            p_on_target = 0.35 + 0.35 * _sigmoid((att_shot - 60) / 20)
            p_on_target = float(np.clip(p_on_target, 0.20, 0.75))

            roll = random()
            if roll < p_on_target:
                # в створ
                stats[side]['on_target'] += 1
                events.append({
                    'minute': minute, 'type': 'shot_on',
                    'team': side,
                    'shooter': att['Имя_чистое'],
                    'xg_shot': att_shot / 100.0,   # 0.5..0.9 примерно
                    'detail': f"{att['Имя_чистое']} бьёт в створ ({gk_name})"
                })
            elif roll < p_on_target + 0.35:
                # угловой
                stats[side]['corners'] += 1
                events.append({
                    'minute': minute, 'type': 'corner',
                    'team': side,
                    'detail': 'угловой'
                })
            else:
                # мимо
                events.append({
                    'minute': minute, 'type': 'miss',
                    'team': side,
                    'detail': f"{att['Имя_чистое']} пробил мимо"
                })

    # =====================================================
    # РЕАЛИЗАЦИЯ ГОЛОВ
    # =====================================================
    # Для каждого удара в створ считаем вероятность гола от xg_shot и вратаря.
    # Углы считаем отдельно.

    def compute_goal_events(side):
        lineup_def = lineup_away if side == 'home' else lineup_home
        gk = pick_gk(lineup_def)
        gk_rating = float(gk['ОР']) if gk is not None else 50.0
        gk_perks = _parse_perks(gk.get('Перки')) if gk is not None else []
        gk_save = _perk_sum(gk_perks, 'save')

        goal_events = []
        for i, e in enumerate(events):
            if e['team'] != side:
                continue
            if e['type'] == 'shot_on':
                # базовое качество удара
                xg_shot = e.get('xg_shot', 0.6)
                # 10% + 15% × (xg_shot - 0.5) + бонус от силы игрока
                base = 0.10 + 0.30 * (xg_shot - 0.5)
                # штраф от вратаря
                base *= (1 - min(0.30, gk_save))
                base *= (1 - min(0.20, (gk_rating - 70) / 100) if gk_rating > 70 else 1.0)
                base = float(np.clip(base, 0.03, 0.45))
                if random() < base:
                    goal_events.append(i)
            elif e['type'] == 'corner':
                # угловой: ~4% гол
                if random() < 0.04:
                    goal_events.append(i)
        return goal_events

    home_goal_idx = compute_goal_events('home')
    away_goal_idx = compute_goal_events('away')

    for idx in home_goal_idx + away_goal_idx:
        e = events[idx]
        side = e['team']
        lineup_att = lineup_home if side == 'home' else lineup_away
        if e['type'] == 'shot_on':
            scorer_name = e['shooter']
        else:
            tall_cands = sorted(
                [p for pos, plist in lineup_att.items()
                 for p in plist if pos in ('ЦЗ', 'ЦН', 'ОФ', 'ОПЗ')],
                key=lambda p: -float(p['Верх'])
            )
            scorer_name = tall_cands[0]['Имя_чистое'] if tall_cands else 'игрок'
        assister_name = None
        if random() < CONFIG['assist_chance']:
            assister = pick_assist(lineup_att, scorer_name)
            if assister is not None:
                assister_name = assister['Имя_чистое']
        events[idx] = {
            'minute': e['minute'],
            'type': 'goal',
            'team': side,
            'scorer': scorer_name,
            'assister': assister_name,
            'detail': 'гол' if assister_name else 'индивидуальный гол',
        }

    score['home'] = sum(1 for i in home_goal_idx)
    score['away'] = sum(1 for i in away_goal_idx)
    # target_home_goals / target_away_goals оставляем такими, какими
    # их разыграл Пуассон — они показывают "ожидаемый" счёт, а не итоговый.
    # Обновляем статистику голов с пенальти/штрафных/угловых
    # (они уже были добавлены в events при эпизодах — оставим как есть)
    abs_diff = abs(or_diff_eff)
    threshold = CONFIG['draw_threshold']
    draw_max = CONFIG['draw_max_chance']
    draw_min = CONFIG['draw_min_chance']
    score_gap = abs(score['home'] - score['away'])

    if score['home'] != score['away'] and score_gap <= CONFIG['draw_score_gap_max']:
        t = max(0.0, 1.0 - abs_diff / threshold)
        p_draw = draw_min + (draw_max - draw_min) * t
        if random() < p_draw:
            m = min(score['home'], score['away'])
            removed_home = score['home'] - m
            removed_away = score['away'] - m

            goals_home_idx = [i for i, e in enumerate(events)
                              if e['type'] == 'goal' and e['team'] == 'home']
            goals_away_idx = [i for i, e in enumerate(events)
                              if e['type'] == 'goal' and e['team'] == 'away']
            to_remove = set()
            for i in goals_home_idx[-removed_home:]:
                to_remove.add(i)
            for i in goals_away_idx[-removed_away:]:
                to_remove.add(i)
            events = [e for i, e in enumerate(events) if i not in to_remove]

            stats['home']['on_target'] = max(0, stats['home']['on_target'] - removed_home)
            stats['away']['on_target'] = max(0, stats['away']['on_target'] - removed_away)

            total_removed = removed_home + removed_away
            if total_removed > 0:
                factor = 1.0 - min(0.5, total_removed * 0.15)
                stats['home']['shots'] = int(stats['home']['shots'] * factor)
                stats['away']['shots'] = int(stats['away']['shots'] * factor)
                stats['home']['corners'] = int(stats['home']['corners'] * factor)
                stats['away']['corners'] = int(stats['away']['corners'] * factor)

            score['home'] = m
            score['away'] = m

    # ---- нормализация владения ----
    total_eps = possession_episodes['home'] + possession_episodes['away']
    if total_eps > 0:
        stats['home']['possession'] = int(round(possession_episodes['home'] / total_eps * 100))
        stats['away']['possession'] = 100 - stats['home']['possession']
    else:
        stats['home']['possession'] = 50
        stats['away']['possession'] = 50

    events.sort(key=lambda e: e['minute'])

    p_home_win = 1.0 / (1.0 + 10 ** (-or_diff_eff / 40))
    xg_home_real = stats['home']['on_target'] * 0.14 + stats['home']['corners'] * 0.04
    xg_away_real = stats['away']['on_target'] * 0.14 + stats['away']['corners'] * 0.04
    return {
        'score': score,
        'events': events,
        'stats': stats,
        'lineups': {'home': lineup_home, 'away': lineup_away},
        'fatigue': fatigue,
        'strength': {'home': S_home_pre, 'away': S_away_pre},
        'expected': {
            'or_home': OR_home,
            'or_away': OR_away,
            'or_diff': or_diff_raw,
            'or_diff_eff': or_diff_eff,
            'home_strength_mult': home_strength_mult,
            'away_strength_mult': away_strength_total,
            'home_or_bonus': home_or_bonus,
            'xg_home': xg_home,              # теоретический xG (для инфо)
            'xg_away': xg_away,
            'target_home_goals': target_home_goals,
            'target_away_goals': target_away_goals,
            'xg_home_real': xg_home_real,
            'xg_away_real': xg_away_real,
            'p_home_win_logit': p_home_win,
            'p_home_poss_avg': p_home_poss_avg,
            'home_adv_on': home_adv_on,
            'home_adv_strength': home_adv_strength,
            'super_home': super_home,
            'super_away': super_away,
            'motivation_home': motivation_home,
            'motivation_away': motivation_away,
            'legendary_match': legendary_match,
            'legendary_team': legendary_team,
        },
    }

# =========================================================
# СЕРИЯ ПЕНАЛЬТИ
# =========================================================
def simulate_penalty_shootout(lineup_home, coach_home,
                              lineup_away, coach_away,
                              seed_val=None,
                              n_shots=5):
    if seed_val is not None:
        np.random.seed(seed_val)

    def shooters(lineup):
        cands = []
        for pos in ('ЦН', 'ОФ', 'АПЗ', 'ЛПЗ', 'ППЗ', 'ЦПЗ', 'ОПЗ'):
            for p in lineup.get(pos, []):
                cands.append(p)
        cands.sort(key=lambda p: -(float(p['Удар']) + float(p['Тактика'])
                                   + float(p['Аура']) * 0.5))
        return cands

    shooters_home = shooters(lineup_home)
    shooters_away = shooters(lineup_away)

    gk_home_l = lineup_home.get('В', [])
    gk_away_l = lineup_away.get('В', [])
    gk_home = gk_home_l[0] if gk_home_l else None
    gk_away = gk_away_l[0] if gk_away_l else None

    def p_score(shooter, gk):
        sh = float(shooter['Удар'])
        tactics = float(shooter['Тактика'])
        if gk is not None:
            g = float(gk['ОР'])
            gk_perks = _parse_perks(gk.get('Перки'))
            save_bonus = _perk_sum(gk_perks, 'save')
        else:
            g = 50.0
            save_bonus = 0.0
        base = 0.75 + 0.10 * _sigmoid((sh + tactics * 0.3 - g) / 20)
        base *= (1 - min(0.25, save_bonus))
        perks = _parse_perks(shooter.get('Перки'))
        base *= (1 + _perk_sum(perks, 'goal') * 0.5)
        return float(np.clip(base, 0.45, 0.95))

    shots = []
    home_score = 0
    away_score = 0

    for i in range(n_shots):
        if i < len(shooters_home):
            sh = shooters_home[i]
            p = p_score(sh, gk_away)
            scored = random() < p
            home_score += int(scored)
            shots.append({'team': 'home', 'order': i + 1,
                          'scorer': sh['Имя_чистое'],
                          'scored': scored, 'probability': round(p, 2)})
        if i < len(shooters_away):
            sh = shooters_away[i]
            p = p_score(sh, gk_home)
            scored = random() < p
            away_score += int(scored)
            shots.append({'team': 'away', 'order': i + 1,
                          'scorer': sh['Имя_чистое'],
                          'scored': scored, 'probability': round(p, 2)})

    max_rounds = 20
    r = 0
    while home_score == away_score and r < max_rounds:
        idx = n_shots + r
        if idx < len(shooters_home):
            sh = shooters_home[idx]
            p = p_score(sh, gk_away)
            scored = random() < p
            home_score += int(scored)
            shots.append({'team': 'home', 'order': idx + 1,
                          'scorer': sh['Имя_чистое'],
                          'scored': scored, 'probability': round(p, 2)})
        if idx < len(shooters_away):
            sh = shooters_away[idx]
            p = p_score(sh, gk_home)
            scored = random() < p
            away_score += int(scored)
            shots.append({'team': 'away', 'order': idx + 1,
                          'scorer': sh['Имя_чистое'],
                          'scored': scored, 'probability': round(p, 2)})
        r += 1

    if home_score > away_score:
        winner = 'home'
    elif away_score > home_score:
        winner = 'away'
    else:
        winner = None

    return {
        'home_score': home_score,
        'away_score': away_score,
        'winner': winner,
        'shots': shots,
    }