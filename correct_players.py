import pandas as pd
from random import choice
path_df = 'players.xlsx'
df = pd.read_excel(path_df,sheet_name='Игроки')



def add_spec(row):
    if row['позиция'] == 'В':
        return 'В'
    if row['позиция'] == 'Т':
        return 'Т'   
    if row['позиция'] == 'З':
        if row['Скорость'] > row['Верх']:
            return choice(['ЛЗ', 'ПЗ'])
        else:
            return 'ЦЗ'
    
    if row['позиция'] == 'П':
        stats = {
            'Скорость': row['Скорость'],
            'Удар':     row['Удар'],
            'Пас':      row['Пас'],
            'Отбор':    row['Отбор'],
            'Верх':     row['Верх'],
            'Тактика':  row['Тактика'],
        }
        best = max(stats, key=stats.get)  # имя поля с максимальным значением
        
        if best == 'Скорость':
            return choice(['ЛПЗ', 'ППЗ'])
        elif best == 'Отбор':
            return 'ОПЗ'
        elif best == 'Пас':
            return 'ЦПЗ'
        elif best == 'Удар':
            return choice(['ЦПЗ', 'АПЗ'])
        elif best == 'Верх':
            return choice(['ЦПЗ', 'ОПЗ'])
        elif best == 'Тактика':
            return choice(['ЦПЗ', 'ОПЗ', 'АПЗ'])
    
    if row['позиция'] == 'Н':
        if row['Пас'] > row['Удар']:
            return 'ОФ'
        else:
            return 'ЦН'
from random import choice, random, shuffle

# -------- совместимые специализации --------
# для каждой базовой позиции — список возможных "допов"
COMPAT = {
    # Защитники
    'ЛЗ': ['ПЗ', 'ЦЗ', 'ОПЗ'],
    'ПЗ': ['ЛЗ', 'ЦЗ', 'ОПЗ'],
    'ЦЗ': ['ОПЗ', 'ЛЗ', 'ПЗ'],
    'ОПЗ': ['ЦЗ'],

    # Полузащитники
    'ЛПЗ': ['ППЗ', 'ЦПЗ', 'ОПЗ', 'АПЗ'],
    'ППЗ': ['ЛПЗ', 'ЦПЗ', 'ОПЗ', 'АПЗ'],
    'ЦПЗ': ['ОПЗ', 'АПЗ', 'ЛПЗ', 'ППЗ'],
    'ОПЗ': ['ЦПЗ', 'ЦЗ'],
    'АПЗ': ['ЦПЗ', 'ЦН', 'ЛПЗ', 'ППЗ'],

    # Нападающие
    'ЦН': ['ОФ', 'АПЗ'],
    'ОФ': ['ЦН', 'АПЗ', 'ЛПЗ', 'ППЗ'],
}

# -------- сколько специализаций давать --------
def n_specs(pos, base, op):
    """
    Возвращает, сколько всего специализаций (включая базовую) будет у игрока.
    Опирается на позицию и ОР.
    """
    r = random()

    if pos in ('В', 'Т'):
        return 1

    if pos == 'З':
        # Защитники: часто 1-2, реже 3
        if op >= 80:
            return 3 if r < 0.35 else 2
        elif op >= 65:
            return 2 if r < 0.55 else 1
        else:
            return 1 if r < 0.7 else 2

    if pos == 'П':
        # Полузащитники: чаще 2-3
        if op >= 80:
            return 3 if r < 0.65 else 2
        elif op >= 65:
            return 2 if r < 0.7 else (3 if r < 0.85 else 1)
        else:
            return 1 if r < 0.5 else 2

    if pos == 'Н':
        # Нападающие: 1-2 обычно, 3 — редкость
        if op >= 80:
            return 2 if r < 0.7 else (3 if r < 0.9 else 1)
        elif op >= 65:
            return 2 if r < 0.5 else 1
        else:
            return 1 if r < 0.75 else 2

    return 1


def pick_specs(base, count):
    """
    Возвращает список специализаций длиной count (включая base).
    """
    if count <= 1:
        return [base]
    candidates = COMPAT.get(base, [])
    # перемешаем и возьмём нужное количество
    pool = candidates[:]
    shuffle(pool)
    result = [base] + pool[:count - 1]
    return result


def add_specs(row):
    """
    Заполняет 3 поля: спец1, спец2, спец3.
    """
    base = add_spec(row)                 # ваша исходная функция
    op = row.get('ОР', 50)               # если столбца нет — дефолт
    n = n_specs(row['позиция'], base, op)
    specs = pick_specs(base, n)

    # добиваем пустыми строками, чтобы всегда было 3 значения
    while len(specs) < 3:
        specs.append('')

    return specs[0], specs[1], specs[2]
df[['спец1', 'спец2', 'спец3']] = df.apply(
    lambda r: pd.Series(add_specs(r)), axis=1
)
df.info()
print(df.sample(5))


with pd.ExcelWriter(path_df, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='total', index=True)
    worksheet = writer.sheets['total']
    workbook = writer.book
    
    # Настройка ширины для колонки индекса
    worksheet.set_column(0, 0, 10)
    
    # Настройка ширины для остальных колонок
    for i, col in enumerate(df.columns):
        col_idx = i + 1  # Исправлено: i + 1, так как колонка индекса - это 0
        width = max(df[col].apply(lambda x: len(str(x))).max(), len(col))
        worksheet.set_column(col_idx, col_idx, width)