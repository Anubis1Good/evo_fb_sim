# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import main as M
import match_sim

st.set_page_config(layout='wide', page_title='Составы')

# ---------- CSS: растягиваем контейнер ----------
st.markdown("""
<style>
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}
.js-plotly-plot, .plot-container {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# КЭШ
# =========================================================
@st.cache_data(show_spinner=False)
def get_base_df():
    return M.load_data()

@st.cache_data(show_spinner=False)
def get_schemas():
    return M.load_schemas()

@st.cache_data(show_spinner=False)
def cached_build_team(kind, name, readiness_hash):
    df = st.session_state.df
    team_full = M.get_team(df, kind, name)
    if len(team_full) == 0:
        return None

    # Отделяем тренера от полевых
    coaches = team_full[team_full['позиция'] == 'Т']
    coach = coaches.sort_values('ОР', ascending=False).iloc[0] if len(coaches) else None
    field = team_full[team_full['позиция'] != 'Т'].copy()

    field = M.apply_coach_to_team(field, coach)
    # Добиваем только полевых (тренера уже выделили)
    field = M.augment_with_phantoms(field, max_total=22, penalty=50.0)

    schemas = get_schemas()
    schema_name, _ = M.pick_best_schema(field, schemas)
    vec = schemas[schema_name]
    lineup, total, used = M.pick_lineup(field, vec)
    bench, _coach2 = M.pick_bench(field, vec, used)

    return dict(
        team=field, schema=schema_name, vec=vec,
        lineup=lineup, total=total, bench=bench, coach=coach,
    )

# =========================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# =========================================================
if 'df' not in st.session_state:
    st.session_state.df = M.recalc_readiness(get_base_df())

if 'match_result' not in st.session_state:
    st.session_state.match_result = None

if 'penalty_result' not in st.session_state:
    st.session_state.penalty_result = None

if 'prev_pair' not in st.session_state:
    st.session_state.prev_pair = None

schemas = get_schemas()
df = st.session_state.df

# =========================================================
# КООРДИНАТЫ ПОЛЯ
# =========================================================
POS_COORDS_REL = {
    'В':   (0.06, 0.50),
    'ЦЗ':  (0.28, 0.50),
    'ЛЗ':  (0.28, 0.15),
    'ПЗ':  (0.28, 0.85),
    'ОПЗ': (0.48, 0.50),
    'ЦПЗ': (0.62, 0.50),
    'ЛПЗ': (0.72, 0.15),
    'ППЗ': (0.72, 0.85),
    'АПЗ': (0.72, 0.50),
    'ОФ':  (0.85, 0.50),
    'ЦН':  (0.97, 0.50),
}

def rel_to_abs(x_rel, y, half):
    if half == 'home':
        return x_rel * 0.5, y
    return 1.0 - x_rel * 0.5, y

def spread_group(items, base_y, dy=0.28):
    n = len(items)
    if n == 0:
        return []
    if n == 1:
        return [(items[0][0], items[0][1], base_y)]
    ys = np.linspace(base_y - dy * (n - 1) / 2,
                     base_y + dy * (n - 1) / 2, n)
    return [(items[i][0], items[i][1], ys[i]) for i in range(n)]

def lineup_to_points(lineup, half):
    groups = {}
    for pos, plist in lineup.items():
        groups.setdefault(pos, []).extend(plist)

    points = []
    for pos, plist in groups.items():
        bx_rel, by = POS_COORDS_REL.get(pos, (0.5, 0.5))
        rows = spread_group([(pos, p) for p in plist], by)
        for _, p, y in rows:
            x, y_abs = rel_to_abs(bx_rel, y, half=half)
            points.append((x, y_abs, pos, p))
    return points

# =========================================================
# РИСОВАНИЕ ПОЛЯ
# =========================================================
def _stat_line(p):
    base = {s: float(p.get(f'{s}_база', p[s])) for s in M.STATS}
    eff  = {s: float(p[s]) for s in M.STATS}
    dom = M.STATS[M._dominant_idx(p)]
    parts = []
    for s in M.STATS:
        if s == dom:
            parts.append(f"{s}: {eff[s]:.0f} (дом)")
        else:
            parts.append(f"{s}: {base[s]:.0f}→{eff[s]:.0f}")
    return '<br>'.join(parts)

def draw_pitch(lineup_home, lineup_away, title=''):
    fig = go.Figure()

    fig.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='white', width=2),
                  fillcolor='#1e5631', layer='below')
    fig.add_shape(type='line', x0=0.5, y0=0, x1=0.5, y1=1,
                  line=dict(color='white', width=2))
    fig.add_shape(type='circle',
                  x0=0.5 - 0.10, y0=0.5 - 0.10,
                  x1=0.5 + 0.10, y1=0.5 + 0.10,
                  line=dict(color='white', width=2))
    fig.add_shape(type='rect', x0=0.0, y0=0.20, x1=0.17, y1=0.80,
                  line=dict(color='white', width=2))
    fig.add_shape(type='rect', x0=0.0, y0=0.36, x1=0.06, y1=0.64,
                  line=dict(color='white', width=2))
    fig.add_shape(type='rect', x0=0.83, y0=0.20, x1=1.0, y1=0.80,
                  line=dict(color='white', width=2))
    fig.add_shape(type='rect', x0=0.94, y0=0.36, x1=1.0, y1=0.64,
                  line=dict(color='white', width=2))

    for lineup, half, color in [(lineup_home, 'home', '#1f6feb'),
                                 (lineup_away, 'away', '#d13438')]:
        for x, y, pos, p in lineup_to_points(lineup, half=half):
            name = p['Имя_чистое']
            rating = float(p.get('ОР_эфф', p['ОР']))
            base_rating = float(p['ОР'])
            ready = int(p['Готовность'])
            perk = M.perk_display(p['Перки'])

            hover = (f"<b>{name}</b><br>"
                    f"Позиция: {pos}<br>"
                    f"ОР: {base_rating:.1f} → {rating:.1f}<br>"
                    f"Готовность: {ready:+d}"
                    + (f"<br>Перк: {perk}" if perk else "")
                    + "<br><br>" + _stat_line(p))

            label = f"<b>{name}</b><br>{rating:.0f} ({ready:+d})"
            if perk:
                label += f"<br>{perk}"

            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(
                    size=64,
                    color=color,
                    line=dict(color='#0b0b0b', width=3),
                ),
                text=[label],
                textposition='middle center',
                textfont=dict(size=12, color='#ffe066', family='Arial'),
                hovertext=[hover], hoverinfo='text',
                showlegend=False
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=26, color='white')),
        xaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False,
                   showticklabels=False, visible=False, fixedrange=True),
        yaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False,
                   showticklabels=False, visible=False, fixedrange=True),
        plot_bgcolor='#1e5631',
        paper_bgcolor='#0e1a12',
        autosize=True,
        height=900,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

# =========================================================
# ТАБЛИЦЫ
# =========================================================
def lineup_to_records(lineup):
    rows = []
    for pos, plist in lineup.items():
        for pl in plist:
            boost = float(pl.get('Тренер_буст', 0.0))
            base_or = float(pl['ОР'])
            eff_or = float(pl.get('ОР_эфф', base_or))
            rows.append(dict(
                Поз=pos,
                Имя=pl['Имя_чистое'],
                ОР=round(base_or, 1),
                ОР_эфф=round(eff_or, 1),
                Буст=f"+{boost*100:.1f}%" if boost > 0 else "—",
                Гот=int(pl['Готовность']),
                Перк=M.perk_display(pl['Перки']),
            ))
    return rows

def bench_to_records(bench):
    rows = []
    for pos, pl in bench:
        rows.append(dict(
            Поз=pos,
            Имя=pl['Имя_чистое'],
            ОР=round(float(pl['ОР']), 1),
            Гот=int(pl['Готовность']),
            Перк=M.perk_display(pl['Перки']),
        ))
    return rows

def coach_line(coach, prof=None):
    if coach is None:
        return None
    name = coach['Имя_чистое']
    rating = float(coach['ОР'])
    ready = int(coach['Готовность'])
    perk = M.perk_display(coach['Перки'])
    s = f"**Тренер:** {name} — ОР {rating:.1f} ({ready:+d})"
    if perk:
        s += f" {perk}"
    if prof is not None:
        doms = sorted(M.STATS, key=lambda x: -prof[x])[:3]
        best = ', '.join(f"{d} +{prof[d]*100:.0f}%" for d in doms if prof[d] > 0)
        if best:
            s += f"\n\n**Сильные стороны:** {best}"
    return s

# =========================================================
# UI
# =========================================================
st.title('⚽ Составы команд')

colA, colB, _ = st.columns([1, 1, 4])
with colA:
    if st.button('🎲 Пересчитать готовность'):
        st.session_state.df = M.recalc_readiness(get_base_df())
        cached_build_team.clear()
        st.session_state.match_result = None
        st.session_state.penalty_result = None
        st.rerun()
with colB:
    if st.button('🔁 Пересчитать схемы'):
        cached_build_team.clear()
        st.session_state.match_result = None
        st.session_state.penalty_result = None
        st.rerun()

# =========================================================
# ВЫБОР КОМАНД
# =========================================================
left, right = st.columns(2)

with left:
    st.subheader('🏠 Домашняя')
    kind_home = st.radio('Тип', ['сборная', 'клуб'],
                         key='kind_home', horizontal=True)
    if kind_home == 'сборная':
        names_home = sorted(df['сборная'].dropna().unique())
    else:
        names_home = sorted(df['клуб_название'].dropna().unique())
    team_home = st.selectbox('Команда', names_home, key='team_home')

with right:
    st.subheader('✈️ Гостевая')
    kind_away = st.radio('Тип ', ['сборная', 'клуб'],
                         key='kind_away', horizontal=True)
    if kind_away == 'сборная':
        names_away = sorted(df['сборная'].dropna().unique())
    else:
        names_away = sorted(df['клуб_название'].dropna().unique())
    team_away = st.selectbox('Команда ', names_away, key='team_away')

# сброс результатов при смене пары
current_pair = (kind_home, team_home, kind_away, team_away)
if st.session_state.prev_pair != current_pair:
    st.session_state.prev_pair = current_pair
    st.session_state.match_result = None
    st.session_state.penalty_result = None

def team_readiness_hash(kind, name):
    t = M.get_team(st.session_state.df, kind, name)
    if len(t) == 0:
        return 'empty'
    return str(int(t['Готовность'].sum()) * 1000 + len(t))

res_home = cached_build_team(kind_home, team_home,
                             team_readiness_hash(kind_home, team_home))
res_away = cached_build_team(kind_away, team_away,
                             team_readiness_hash(kind_away, team_away))

# =========================================================
# КНОПКИ СИМУЛЯЦИИ + ДОМАШНЕЕ ПРЕИМУЩЕСТВО
# =========================================================
if res_home and res_away:
    st.markdown('---')
    st.markdown('### ⚔️ Симуляция матча')

    colH1, colH2 = st.columns([1, 3])
    with colH1:
        home_adv_on = st.checkbox('🏟 Домашнее преимущество', value=True)
    with colH2:
        home_adv_strength = st.slider(
            'Сила домашнего преимущества', 0.0, 2.0, 1.0, 0.1,
            disabled=not home_adv_on,
        )

    colM1, colM2, colM3 = st.columns([1, 1, 2])
    with colM1:
        if st.button('⚔️ Симулировать матч'):
            st.session_state.match_result = match_sim.simulate_match(
                res_home['lineup'], res_home['coach'],
                res_away['lineup'], res_away['coach'],
                bench_home=res_home['bench'],
                bench_away=res_away['bench'],
                home_adv_on=home_adv_on,
                home_adv_strength=home_adv_strength,
            )
            st.session_state.penalty_result = None
    with colM2:
        if st.button('🎯 Серия пенальти'):
            st.session_state.penalty_result = match_sim.simulate_penalty_shootout(
                res_home['lineup'], res_home['coach'],
                res_away['lineup'], res_away['coach'],
            )
            st.session_state.match_result = None

    # =========================================================
    # РЕЗУЛЬТАТ МАТЧА
    # =========================================================
    result = st.session_state.get('match_result')
    if result:
        sh, sa = result['score']['home'], result['score']['away']
        st.markdown(f"## 🏁 {team_home} **{sh} : {sa}** {team_away}")

        exp = result.get('expected', {})
        if exp:
            st.caption(
                f"ОР: {exp['or_home']:.1f} vs {exp['or_away']:.1f}  "
                f"(Δ = {exp['or_diff']:+.1f}, с домом {exp['or_diff_eff']:+.1f})  |  "
                f"xG: {exp['xg_home']:.2f} vs {exp['xg_away']:.2f}  |  "
                f"ожидаемый счёт (Пуассон): {exp['target_home_goals']}:{exp['target_away_goals']}  |  "
                f"xG: {exp['xg_home']:.2f} vs {exp['xg_away']:.2f}  "
f"(реальный {exp.get('xg_home_real', 0):.2f} vs {exp.get('xg_away_real', 0):.2f})  |  "
                f"P(победа хозяев) ≈ {exp['p_home_win_logit']:.2f}  |  "
                f"дом: {'вкл' if exp['home_adv_on'] else 'выкл'} "
                f"(×{exp['home_strength_mult']:.2f}, Δ+{exp['home_or_bonus']:+.1f})"
            )
        if exp.get('legendary_match'):
            lt = exp.get('legendary_team')
            team_name = team_home if lt == 'home' else team_away
            st.markdown('# 🌟 ЛЕГЕНДАРНЫЙ МАТЧ')
            st.markdown(f'**{team_name} сегодня играет на пределе возможностей (+50%).**')
        if exp.get('super_home'):
            st.markdown('🔥 **Супермотивация хозяев** (+25%)')
        if exp.get('super_away'):
            st.markdown('🔥 **Супермотивация гостей** (+25%)')
        stats = result['stats']
        st.markdown(
            f"**Владение:** {stats['home']['possession']}% — "
            f"{stats['away']['possession']}%  |  "
            f"**Удары:** {stats['home']['shots']} — {stats['away']['shots']}  |  "
            f"**В створ:** {stats['home']['on_target']} — {stats['away']['on_target']}  |  "
            f"**Углы:** {stats['home']['corners']} — {stats['away']['corners']}  |  "
            f"**Фолы:** {stats['home']['fouls']} — {stats['away']['fouls']}  |  "
            f"**ЖК:** {stats['home']['yellow']} — {stats['away']['yellow']}  |  "
            f"**КК:** {stats['home']['red']} — {stats['away']['red']}  |  "
            f"**Пенальти:** {stats['home']['penalties']} — {stats['away']['penalties']} "
            f"(забито {stats['home']['goals_from_pen']} — {stats['away']['goals_from_pen']})  |  "
            f"**Травмы:** {stats['home']['injuries']} — {stats['away']['injuries']}"
        )

        # ---- ГОЛЫ ----
        goals = [e for e in result['events'] if e['type'] == 'goal']
        st.markdown('### ⚽ Голы')
        if goals:
            for e in goals:
                team_name = team_home if e['team'] == 'home' else team_away
                assister = e.get('assister')
                assist_str = f"  (пас: {assister})" if assister else "  (без передачи)"
                st.markdown(
                    f"**{e['minute']}'** — {team_name}: "
                    f"**{e['scorer']}**{assist_str}  —  _{e.get('detail','')}_"
                )
        else:
            st.markdown('_Без голов._')

        # ---- НЕРЕАЛИЗОВАННЫЕ ПЕНАЛЬТИ ----
        pen_miss = [e for e in result['events'] if e['type'] == 'penalty_miss']
        if pen_miss:
            st.markdown('### ❌ Нереализованные пенальти')
            for e in pen_miss:
                team_name = team_home if e['team'] == 'home' else team_away
                st.markdown(f"**{e['minute']}'** — {team_name}: {e['detail']}")

        # ---- ТАБЛИЦА ГОЛОВ И АССИСТОВ ----
        st.markdown('### 📋 Голы и голевые передачи')
        rows = []
        for e in result['events']:
            if e['type'] != 'goal':
                continue
            team_name = team_home if e['team'] == 'home' else team_away
            rows.append({
                'Минута': e['minute'],
                'Команда': team_name,
                'Игрок': e['scorer'],
                'Тип': 'Гол',
                'Ассистент': e.get('assister') or '—',
                'Деталь': e.get('detail', ''),
            })
            if e.get('assister'):
                rows.append({
                    'Минута': e['minute'],
                    'Команда': team_name,
                    'Игрок': e['assister'],
                    'Тип': 'Ассист',
                    'Ассистент': e['scorer'],
                    'Деталь': e.get('detail', ''),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows),
                         hide_index=True, use_container_width=True)
        else:
            st.markdown('_Нет голов и передач._')

        # ---- ЗАМЕНЫ ----
        subs = [e for e in result['events']
                if e['type'] in ('substitution', 'injury_sub')]
        if subs:
            st.markdown('### ↔ Замены')
            for e in subs:
                team_name = team_home if e['team'] == 'home' else team_away
                st.markdown(f"**{e['minute']}'** — {team_name}: {e['detail']}")

        # ---- ТРАВМЫ ----
        injs = [e for e in result['events']
                if e['type'] in ('injury_sub', 'injury_no_sub')]
        if injs:
            st.markdown('### 🚑 Травмы')
            for e in injs:
                team_name = team_home if e['team'] == 'home' else team_away
                st.markdown(f"**{e['minute']}'** — {team_name}: {e['detail']}")

        # ---- КАРТОЧКИ ----
        cards = [e for e in result['events']
                 if e['type'] in ('yellow', 'red')]
        if cards:
            st.markdown('### 🟨🟥 Карточки')
            for e in cards:
                team_name = team_home if e['team'] == 'home' else team_away
                emoji = '🟨' if e['type'] == 'yellow' else '🟥'
                st.markdown(f"**{e['minute']}'** — {team_name}: {emoji} {e['detail']}")

        # ---- УСТАЛОСТЬ ----
        if 'fatigue' in result:
            fat = result['fatigue']
            nonzero = {k: v for k, v in fat.items() if v > 0.5}
            if nonzero:
                st.markdown('### 😮 Усталость (конец матча)')
                fat_rows = [{'Игрок': name, 'Усталость': round(v, 1)}
                            for name, v in sorted(nonzero.items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(fat_rows),
                             hide_index=True, use_container_width=True)

        # ---- ВСЕ СОБЫТИЯ ----
        with st.expander('Все события матча'):
            for e in result['events']:
                team_name = team_home if e['team'] == 'home' else team_away
                line = f"{e['minute']:>3}'  [{team_name}]  "
                if e['type'] == 'shot_on':
                    line += f"🥅 {e['detail']}"
                elif e['type'] == 'goal':
                    assist = f" (пас: {e['assister']})" if e.get('assister') else ""
                    line += f"⚽ ГОЛ — {e['scorer']}{assist}  — {e.get('detail','')}"
                elif e['type'] == 'yellow':
                    line += f"🟨 ЖК — {e['detail']}"
                elif e['type'] == 'red':
                    line += f"🟥 КК — {e['detail']}"
                elif e['type'] == 'save':
                    line += f"🧤 {e['detail']}"
                elif e['type'] == 'corner':
                    line += "🚩 угловой"
                elif e['type'] == 'interception':
                    line += f"🛡 {e['detail']}"
                elif e['type'] == 'miss':
                    line += f"❌ {e['detail']}"
                elif e['type'] in ('substitution', 'injury_sub'):
                    line += f"↔ {e['detail']}"
                elif e['type'] == 'injury_no_sub':
                    line += f"🚑 {e['detail']}"
                elif e['type'] == 'penalty_miss':
                    line += f"❌ {e['detail']}"
                else:
                    line += f"{e['detail']}"
                st.write(line)

    # =========================================================
    # РЕЗУЛЬТАТ ПЕНАЛЬТИ
    # =========================================================
    pres = st.session_state.get('penalty_result')
    if pres:
        st.markdown(
            f"### 🎯 Серия пенальти: {team_home} "
            f"**{pres['home_score']} : {pres['away_score']}** {team_away}"
        )
        if pres['winner'] == 'home':
            st.success(f'Победа {team_home}')
        elif pres['winner'] == 'away':
            st.success(f'Победа {team_away}')
        else:
            st.info('Ничья — sudden death без лимита, быть не должно')

        st.markdown('**Удары:**')
        for s in pres['shots']:
            team_name = team_home if s['team'] == 'home' else team_away
            mark = '✅' if s['scored'] else '❌'
            st.write(
                f"{s['order']:>2}. {mark}  [{team_name}]  "
                f"{s['scorer']}  (шанс {s['probability']:.2f})"
            )

# =========================================================
# ПОЛЕ И СОСТАВЫ
# =========================================================
if res_home and res_away:
    st.markdown('---')
    fig = draw_pitch(
        res_home['lineup'], res_away['lineup'],
        title=(f"{team_home}   {res_home['total']:.1f}   vs   "
               f"{res_away['total']:.1f}   {team_away}")
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False, 'responsive': True},
    )

    c1, c2 = st.columns(2)
    LINEUP_HEIGHT = 42 * 11 + 40
    BENCH_HEIGHT = 42 * 11 + 40

    with c1:
        st.markdown(
            f"### 🏠 {team_home} — схема **{res_home['schema']}** "
            f"(Σ ОР = {res_home['total']:.1f})"
        )
        st.dataframe(
            pd.DataFrame(lineup_to_records(res_home['lineup'])),
            hide_index=True, use_container_width=True,
            height=LINEUP_HEIGHT
        )
        st.markdown('**Запасные:**')
        st.dataframe(
            pd.DataFrame(bench_to_records(res_home['bench'])),
            hide_index=True, use_container_width=True,
            height=BENCH_HEIGHT
        )
        cl = coach_line(res_home['coach'])
        if cl:
            st.markdown(cl)

    with c2:
        st.markdown(
            f"### ✈️ {team_away} — схема **{res_away['schema']}** "
            f"(Σ ОР = {res_away['total']:.1f})"
        )
        st.dataframe(
            pd.DataFrame(lineup_to_records(res_away['lineup'])),
            hide_index=True, use_container_width=True,
            height=LINEUP_HEIGHT
        )
        st.markdown('**Запасные:**')
        st.dataframe(
            pd.DataFrame(bench_to_records(res_away['bench'])),
            hide_index=True, use_container_width=True,
            height=BENCH_HEIGHT
        )
        cl = coach_line(res_away['coach'])
        if cl:
            st.markdown(cl)

else:
    st.warning('В одной из команд нет игроков.')