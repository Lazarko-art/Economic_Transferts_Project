# Глава 3. Симуляция влияния системы социальных трансфертов на доходы
# домохозяйств Республики Беларусь

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

np.random.seed(2025)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SAVE_DIR, "chapter3_plots.png")

# ============================================================
# БЛОК 1: ПАРАМЕТРЫ (БПМ по группам)
# ============================================================

BPM_WORKING    = 554.40   # трудоспособные
BPM_RETIRED    = 366.59   # пенсионеры
BPM_CHILD_0_3  = 314.10   # дети до 3 лет
BPM_CHILD_3_6  = 422.66   # дети 3–6 лет
BPM_CHILD_6_18 = 497.59   # дети 6–18 лет

MEAN_INCOME = 1500        # средний доход на душу по Белстату

n = 2_000

print("=" * 80)
print("СИМУЛЯЦИЯ ВЛИЯНИЯ СИСТЕМЫ СОЦИАЛЬНЫХ ТРАНСФЕРТОВ НА ДОХОДЫ НАСЕЛЕНИЯ")
print("=" * 80)

# ============================================================
# БЛОК 2: ДЕМОГРАФИЯ ДОМОХОЗЯЙСТВ
# ============================================================

family_size = np.random.choice([1,2,3,4,5,6], size=n,
                                p=[0.10,0.25,0.25,0.20,0.12,0.08])

children = np.random.choice([0,1,2,3,4], size=n,
                              p=[0.40,0.28,0.18,0.09,0.05])
children = np.minimum(children, family_size - (family_size > 1).astype(int))
children = np.maximum(children, 0)

retired = np.random.choice([0,1,2], size=n, p=[0.65,0.25,0.10])
retired = np.minimum(retired, family_size - children)

working = np.maximum(family_size - children - retired, 0)

age = (np.random.beta(a=2, b=2, size=n) * 62 + 18).astype(int)

# Образование: 1=среднее, 2=среднее специальное (база), 3=высшее
edu = np.random.choice([1,2,3], size=n, p=[0.30,0.45,0.25])

# Тип поселения: 1=город, 0=село
urban = np.random.choice([0,1], size=n, p=[0.30,0.70])

# Регион: 1=Минск, 2-4=прочие области, 5=сельская область
region = np.random.choice([1,2,3,4,5], size=n, p=[0.20]*5)

has_children = (children > 0).astype(float)
log_fs = np.log(family_size)
log_w1 = np.log1p(working)

# ============================================================
# БЛОК 3: ДОХОД НА ДУШУ (в лог-пространстве)
# ============================================================
# log(доход на душу) = систематическая часть + случайный шум
# Демографическая механика:
#   + log(1 + working): больше занятых → выше заработок семьи
#   - log(family_size): больше членов → ниже доход на душу

SIGMA_EPSILON = 0.22

BETA_TREAT       = 0.10
BETA_TREAT_CHILD = 0.05

systematic = (
    7.50
    + 0.35 * (edu == 3).astype(float)     # премия за высшее образование
    - 0.20 * (edu == 1).astype(float)     # штраф за только среднее
    + 0.008 * (age - 40)                  # рост дохода с опытом
    - 0.0001 * (age - 40) ** 2            # убывающая отдача после пика
    + 0.18 * urban                        # городская надбавка
    + 0.22 * (region == 1).astype(float)  # Минск
    - 0.12 * (region == 5).astype(float)  # сельская область
    + 0.40 * log_w1                       # трудовой потенциал семьи
    - 0.80 * log_fs                       # иждивенческая нагрузка
)

# ============================================================
# БЛОК 4: НАЗНАЧЕНИЕ ТРАНСФЕРТОВ
# ============================================================
# Белорусская система социальных трансфертов назначается
# по демографическим критериям:
#
#   Пособия на детей (Закон № 29-З от 29.12.2012):
#     - при рождении и до 3 лет: 35–45% среднемес. зарплаты
#     - от 3 до 18 лет: 50% БПМ ребёнка соответствующей группы
#     - многодетным: повышенные коэффициенты
#
#   Пенсии по возрасту/инвалидности:
#     - назначаются при достижении пенсионного возраста
#       (63 года муж., 58 лет жен.) независимо от дохода семьи
#
#   Субсидии на оплату ЖКУ:
#     - льготные тарифы для сельского населения
#     - региональные надбавки в отдельных областях

prob_treatment = np.clip(
    0.10
    + 0.30 * (children > 0).astype(float)   # пособия на детей
    + 0.20 * (children >= 2).astype(float)  # повышенное пособие многодетным
    + 0.25 * (retired > 0).astype(float)    # пенсии
    + 0.10 * (1 - urban).astype(float)      # субсидии ЖКУ 
    + 0.05 * (region == 5).astype(float),   # региональные доплаты
    0.02, 0.92
)

treatment = np.random.binomial(1, prob_treatment)

# ============================================================
# БЛОК 5: ФИНАЛЬНЫЙ DGP
# ============================================================
# log(доход_после) = систематическая + эффект трансфертов + шум
# log(доход_до)    = систематическая + шум (εᵢ)
#
# Одинаковый шум ε для обоих уравнений гарантирует, что разница
# между доходом до и после отражает только эффект трансфертов.

epsilon = np.random.normal(0, SIGMA_EPSILON, size=n)

log_income_after  = (systematic
                     + BETA_TREAT       * treatment
                     + BETA_TREAT_CHILD * treatment * has_children
                     + epsilon)

log_income_before = (log_income_after
                     - BETA_TREAT       * treatment
                     - BETA_TREAT_CHILD * treatment * has_children)

income_before = np.exp(log_income_before)
income_after  = np.exp(log_income_after)
subsidy_pc    = np.maximum(income_after - income_before, 0)

# ============================================================
# БЛОК 6: ДАТАФРЕЙМ (таблица данных)
# ============================================================

df = pd.DataFrame({
    'family_size':       family_size,
    'children':          children,
    'retired':           retired,
    'working':           working,
    'age':               age,
    'edu':               edu,
    'urban':             urban,
    'region':            region,
    'income_before':     income_before,
    'income_after':      income_after,
    'log_income_before': log_income_before,
    'log_income_after':  log_income_after,
    'treatment':         treatment,
    'subsidy':           subsidy_pc,
})

df['edu_high']             = (df['edu'] == 3).astype(int)
df['edu_low']              = (df['edu'] == 1).astype(int)
df['age_c']                = df['age'] - 40
df['age_c2']               = df['age_c'] ** 2
df['log_fs']               = log_fs
df['log_w1']               = log_w1
df['region_minsk']         = (df['region'] == 1).astype(int)
df['region_rural']         = (df['region'] == 5).astype(int)
df['treatment_x_children'] = df['treatment'] * (df['children'] > 0).astype(int)

treat_mask = df['treatment'] == 1

# ============================================================
# БЛОК 7: ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ============================================================

print(f"\n=== СТАТИСТИКА ПО ДОХОДАМ НА ДУШУ (ДО ТРАНСФЕРТОВ) ===")
print(f"Средний доход на душу:                {income_before.mean():.0f} руб.")
print(f"Медианный доход на душу:              {np.median(income_before):.0f} руб.")
print(f"Семьи БЕЗ детей (на душу):           {income_before[children==0].mean():.0f} руб.")
print(f"Семьи С детьми  (на душу):           {income_before[children>0].mean():.0f} руб.")
print(f"Доля с доходом < БПМ ({BPM_WORKING} руб.):  {(income_before < BPM_WORKING).mean()*100:.1f}%")

print(f"\n=== ПОЛУЧАТЕЛИ ТРАНСФЕРТОВ ({df['treatment'].sum()} из {n}) ===")
print(f"Средний доход на душу ДО:             {df[treat_mask]['income_before'].mean():.0f} руб.")
print(f"Средний доход на душу ПОСЛЕ:          {df[treat_mask]['income_after'].mean():.0f} руб.")
print(f"Средний трансферт на душу:            {df[treat_mask]['subsidy'].mean():.0f} руб.")
print(f"Неполучатели — доход на душу:         {df[~treat_mask]['income_before'].mean():.0f} руб.")

gap_before = df[~treat_mask]['income_before'].mean() - df[treat_mask]['income_before'].mean()
gap_after  = df[~treat_mask]['income_after'].mean()  - df[treat_mask]['income_after'].mean()
print(f"\nРазрыв ДО трансфертов:   {gap_before:.0f} руб.")
print(f"Разрыв ПОСЛЕ трансфертов: {gap_after:.0f} руб.")
print(f"Трансферты закрыли {(1 - gap_after/gap_before)*100:.0f}% разрыва")

true_att_abs = df[treat_mask]['subsidy'].mean()
true_att_pct = (df[treat_mask]['income_after'].mean() /
                df[treat_mask]['income_before'].mean() - 1) * 100

print(f"\n=== ИСТИННЫЙ ЭФФЕКТ ТРАНСФЕРТОВ (ATT) ===")
print(f"Абсолютный прирост на душу:  +{true_att_abs:.0f} руб.")
print(f"Относительный прирост:       +{true_att_pct:.1f}%")

# ============================================================
# БЛОК 8: НАИВНЫЙ ЭФФЕКТ vs МНК-РЕГРЕССИЯ
# ============================================================
# Наивный эффект — простая разница средних доходов после трансфертов.

naive_effect = (df[treat_mask]['income_after'].mean() -
                df[~treat_mask]['income_after'].mean())
print(f"\n=== НАИВНЫЙ ЭФФЕКТ (сравнение средних после) ===")
print(f"Наивный эффект: {naive_effect:+.0f} руб.")

X_cols = ['treatment', 'treatment_x_children', 'age_c', 'age_c2',
          'edu_high', 'edu_low', 'urban', 'log_w1', 'log_fs',
          'region_minsk', 'region_rural']
X = sm.add_constant(df[X_cols])
y = df['log_income_after']
model = sm.OLS(y, X).fit(cov_type='HC3')

est_log = model.params['treatment']
est_pct = (np.exp(est_log) - 1) * 100

print(f"\n=== РЕГРЕССИОННАЯ ОЦЕНКА (МНК с контролем факторов) ===")
print(model.summary())
print(f"\nОценка эффекта трансфертов: +{est_pct:.1f}% к доходу на душу")
print(f"Смещение относительно истинного ATT: {est_pct - true_att_pct:.1f} п.п.")

residuals = model.resid

# ============================================================
# БЛОК 9: ТЕСТ БРЕУША-ПАГАНА
# ============================================================

resid_sq   = residuals ** 2
X_bp       = sm.add_constant(model.fittedvalues)
bp_model   = sm.OLS(resid_sq, X_bp).fit()
lm_stat    = n * bp_model.rsquared
p_value_bp = 1 - stats.chi2.cdf(lm_stat, df=1)

print("\n" + "=" * 80)
print("ТЕСТ БРЕУША-ПАГАНА (гомоскедастичность остатков)")
print("=" * 80)
print(f"Статистика LM = n·R² = {lm_stat:.4f}")
print(f"p-value: {p_value_bp:.4f}")
print("Вывод: " + ("✅ гетероскедастичность отсутствует"
                    if p_value_bp > 0.05 else "⚠️ присутствует гетероскедастичность"))

# ============================================================
# БЛОК 10: ТЕСТ ЖАРКЕ-БЕРА
# ============================================================

jb_stat, jb_pvalue = stats.jarque_bera(residuals)

print("\n" + "=" * 80)
print("ТЕСТ ЖАРКЕ-БЕРА (нормальность остатков)")
print("=" * 80)
print(f"Статистика JB: {jb_stat:.4f}")
print(f"p-value: {jb_pvalue:.4f}")
print("Вывод: " + ("✅ остатки нормальны"
                    if jb_pvalue > 0.05 else "⚠️ отклонение от нормальности"))

# ============================================================
# БЛОК 11: ТЕСТ УАЙТА
# ============================================================

predicted    = model.fittedvalues
predicted_sq = predicted ** 2
X_white      = sm.add_constant(
    pd.DataFrame({'predicted': predicted, 'predicted_sq': predicted_sq}))
white_model  = sm.OLS(resid_sq, X_white).fit()
white_lm     = n * white_model.rsquared
white_pvalue = 1 - stats.chi2.cdf(white_lm, df=2)

print("\n" + "=" * 80)
print("ТЕСТ УАЙТА (гетероскедастичность)")
print("=" * 80)
print(f"Статистика LM = n·R² = {white_lm:.4f}")
print(f"p-value: {white_pvalue:.4f}")
print("Вывод: " + ("✅ гетероскедастичность отсутствует"
                    if white_pvalue > 0.05 else "⚠️ присутствует гетероскедастичность"))

# ============================================================
# БЛОК 12: МОДЕЛЬ С ВЗАИМОДЕЙСТВИЕМ
# ============================================================

X_int = sm.add_constant(df[['treatment', 'treatment_x_children', 'age_c', 'age_c2',
                              'edu_high', 'edu_low', 'urban', 'log_w1', 'log_fs',
                              'region_minsk', 'region_rural']])
model_int = sm.OLS(df['log_income_after'], X_int).fit(cov_type='HC3')

eff_no_ch   = model_int.params['treatment']
eff_with_ch = eff_no_ch + model_int.params['treatment_x_children']

print("\n" + "=" * 80)
print("МОДЕЛЬ С ВЗАИМОДЕЙСТВИЕМ (гетерогенный эффект)")
print("=" * 80)
print(f"Эффект для семей БЕЗ детей: {(np.exp(eff_no_ch)-1)*100:.1f}%")
print(f"Эффект для семей С детьми:  {(np.exp(eff_with_ch)-1)*100:.1f}%")
print(f"Доп. выигрыш семей с детьми: {((np.exp(eff_with_ch)-1)*100-(np.exp(eff_no_ch)-1)*100):.1f} п.п.")

# ============================================================
# БЛОК 13: ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Симуляция системы социальных трансфертов: диагностика модели",
             fontsize=14, fontweight='bold')

ax = axes[0, 0]
sns.kdeplot(df[df['treatment']==1]['income_before'],
            label='Получатели (до)', fill=True, alpha=0.5, bw_method=0.3, ax=ax)
sns.kdeplot(df[df['treatment']==0]['income_before'],
            label='Неполучатели (до)', fill=True, alpha=0.5, bw_method=0.3, ax=ax)
ax.axvline(BPM_WORKING, color='orange', linestyle=':', lw=1.5,
           label=f'БПМ ({BPM_WORKING} руб.)')
ax.set_title('Доход на душу ДО трансфертов', fontweight='bold')
ax.set_xlabel('Руб./мес.'); ax.set_xlim(0, 3500)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
sns.kdeplot(df[df['treatment']==1]['income_after'],
            label='Получатели (после)', fill=True, alpha=0.5, bw_method=0.3, ax=ax)
sns.kdeplot(df[df['treatment']==0]['income_after'],
            label='Неполучатели (после)', fill=True, alpha=0.5, bw_method=0.3, ax=ax)
ax.axvline(BPM_WORKING, color='orange', linestyle=':', lw=1.5,
           label=f'БПМ ({BPM_WORKING} руб.)')
ax.set_title('Доход на душу ПОСЛЕ трансфертов', fontweight='bold')
ax.set_xlabel('Руб./мес.'); ax.set_xlim(0, 3500)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(model.fittedvalues, residuals, alpha=0.2, s=8, color='steelblue')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
ax.set_title('Остатки vs предсказанные\n(равномерный разброс → гомоскедастичность)',
             fontweight='bold')
ax.set_xlabel('Предсказанные log(доход)'); ax.set_ylabel('Остатки')
ax.grid(alpha=0.3)

ax = axes[1, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, alpha=0.3, s=8, color='steelblue', label='Квантили остатков')
ax.plot(osm, slope*np.array(osm)+intercept, color='red', lw=2, label='Нормаль')
ax.set_title(f'Q-Q график остатков\n'
             f'JB p={jb_pvalue:.3f}, BP p={p_value_bp:.3f}, White p={white_pvalue:.3f}',
             fontweight='bold')
ax.set_xlabel('Теоретические квантили'); ax.set_ylabel('Выборочные квантили')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nГрафики сохранены: {PLOT_PATH}")

# ============================================================
# БЛОК 14: АНАЛИЗ БПМ
# ============================================================

before_bpm = (df[treat_mask]['income_before'] < BPM_WORKING).mean() * 100
after_bpm  = (df[treat_mask]['income_after']  < BPM_WORKING).mean() * 100

print("\n" + "=" * 80)
print(f"АНАЛИЗ УРОВНЯ БПМ ({BPM_WORKING} руб.)")
print("=" * 80)
print(f"Получатели с доходом ниже БПМ ДО:    {before_bpm:.1f}%")
print(f"Получатели с доходом ниже БПМ ПОСЛЕ: {after_bpm:.1f}%")
print(f"Сокращение:                           {before_bpm - after_bpm:.1f} п.п.")

# ============================================================
# БЛОК 15: ИТОГОВЫЕ ВЫВОДЫ
# ============================================================

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ЭКОНОМИЧЕСКИЕ ВЫВОДЫ")
print("=" * 80)
print(f"""
1. ЭКОНОМИЧЕСКАЯ ЛОГИКА ДОХОДА НА ДУШУ:
   - Семьи с детьми беднее на душу при равном заработке взрослых:
     без детей: {income_before[children==0].mean():.0f} руб., с детьми: {income_before[children>0].mean():.0f} руб.
   - Трансферты компенсируют этот разрыв, особенно для многодетных

2. ПОЛУЧАТЕЛИ vs НЕПОЛУЧАТЕЛИ:
   - Получатели (до):    {df[treat_mask]['income_before'].mean():.0f} руб. на душу
   - Неполучатели (до):  {df[~treat_mask]['income_before'].mean():.0f} руб. на душу
   - Получатели (после): {df[treat_mask]['income_after'].mean():.0f} руб. на душу
   - Разрыв сократился с {gap_before:.0f} до {gap_after:.0f} руб. (на {(1-gap_after/gap_before)*100:.0f}%)

3. ДИАГНОСТИКА МОДЕЛИ:
   - Тест Бреуша-Пагана: {'✅ гомоскедастичность' if p_value_bp > 0.05 else '⚠️ гетероскедастичность'} (p={p_value_bp:.4f})
   - Тест Уайта:          {'✅ гомоскедастичность' if white_pvalue > 0.05 else '⚠️ гетероскедастичность'} (p={white_pvalue:.4f})
   - Тест Жарке-Бера:     {'✅ нормальность' if jb_pvalue > 0.05 else '⚠️ ненормальность'} (p={jb_pvalue:.4f})

4. ГЕТЕРОГЕННЫЙ ЭФФЕКТ:
   - Семьи без детей: +{(np.exp(eff_no_ch)-1)*100:.1f}%
   - Семьи с детьми:  +{(np.exp(eff_with_ch)-1)*100:.1f}%
   - Доп. выигрыш семей с детьми: {((np.exp(eff_with_ch)-1)*100-(np.exp(eff_no_ch)-1)*100):.1f} п.п.

5. СНИЖЕНИЕ ДОЛИ НИЖЕ БПМ:
   - Доля получателей ниже БПМ сократилась: {before_bpm:.1f}% → {after_bpm:.1f}%
   - Сокращение: {before_bpm - after_bpm:.1f} п.п.
""")
