"""Static Russian Markdown/HTML report and single-panel Matplotlib figures."""
from __future__ import annotations
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    'random': 'Случайный показ',
    'popularity_public': 'Публичная популярность',
    'popularity_train': 'Популярность по обучающей истории',
    'compatibility_oracle': 'Ориентир по совместимости',
    'mixed_logger': 'Смешанная исходная политика',
    'mf': 'Централизованная матричная факторизация',
}


def _save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(root/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def _bar(aggregate, metric, k, methods, root, filename, title, xlabel):
    subset = aggregate[(aggregate.metric == metric) & (aggregate.k == k)].set_index('method')
    names = [m for m in methods if m in subset.index and np.isfinite(subset.loc[m, 'mean'])]
    if not names:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(names))
    ax.barh(y, [subset.loc[m, 'mean'] for m in names],
            xerr=[subset.loc[m, 'std'] for m in names], capsize=3)
    ax.set_yticks(y, [LABELS[m] for m in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', alpha=.25)
    _save(fig, root, filename)
    return True


def create_report(root: Path, config: dict, aggregate: pd.DataFrame,
                  catalog: pd.DataFrame, training: pd.DataFrame, splits: list[dict]) -> None:
    figures = root/'figures'
    figures.mkdir()
    methods = config['methods']
    max_k = max(config['evaluation']['catalog_k'])
    choice_k = max(config['evaluation']['choice_k'])
    images = []
    fig, ax = plt.subplots(figsize=(9, 5.5))
    subset = aggregate[aggregate.metric.eq('compatibility_at_k')]
    for method in methods:
        points = subset[subset.method.eq(method)].sort_values('k')
        ax.errorbar(points.k, points['mean'], yerr=points['std'], marker='o', capsize=3,
                    label=LABELS[method])
    ax.set(xlabel='Количество рекомендованных стратегий K', ylabel='Средняя совместимость',
           title='Соответствие профилю: оценка по всему каталогу')
    ax.set_xticks(config['evaluation']['catalog_k'])
    ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='best')
    _save(fig, figures, '01_compatibility')
    images.append(('01_compatibility', 'Совместимость: полный каталог, линейное усреднение по пользователям.'))
    if _bar(aggregate, 'oracle_regret_at_k', max_k, methods, figures, '02_oracle_gap',
            f'Разрыв до ориентира по совместимости, K={max_k}', 'Разрыв; меньше — лучше'):
        images.append(('02_oracle_gap', 'Ориентир является верхней границей только для совместимости.'))
    if _bar(aggregate, 'ndcg_at_k', choice_k, methods, figures, '03_heldout_choice',
            f'Ранжирование внутри тестового показа, K={choice_k}', 'NDCG: выше — лучше'):
        images.append(('03_heldout_choice', 'Только тестовые списки с копированием; это предсказание выбора, не оценка новой политики по старым логам.'))
    if _bar(aggregate, 'expected_copy_rate', 0, methods, figures, '04_response',
            'Новые показы при замороженных моделях', 'Средняя условная вероятность хотя бы одного копирования'):
        images.append(('04_response', 'Новые отклики в одинаковых сброшенных мирах; модели после них не дообучаются.'))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    group = (catalog[catalog.k.eq(max_k)]
             .groupby(['seed','method','user_type'], as_index=False).compatibility_at_k.mean()
             .groupby(['method','user_type'], as_index=False).compatibility_at_k.mean())
    types = sorted(group.user_type.unique())
    type_labels = {'conservative':'Консервативные','balanced':'Сбалансированные',
                   'aggressive':'Агрессивные','social_driven':'Ориентированные на популярность'}
    for method in methods:
        values = group[group.method.eq(method)].set_index('user_type').reindex(types)
        ax.plot(np.arange(len(types)), values.compatibility_at_k, marker='o', label=LABELS[method])
    ax.set_xticks(np.arange(len(types)), [type_labels.get(t,t) for t in types], fontsize=8)
    ax.set(ylabel=f'Совместимость@{max_k}', title='Качество по группам пользователей', ylim=(0,1))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    _save(fig, figures, '05_user_groups')
    images.append(('05_user_groups', 'Сначала среднее внутри группы и одного запуска, затем среднее по независимым мирам.'))
    if not training.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for seed, h in training.groupby('seed', sort=True):
            ax.plot(h.epoch, h.train_bce, label=f'Обучение, seed={seed}')
            ax.plot(h.epoch, h.validation_bce, linestyle='--', label=f'Проверка, seed={seed}')
            best = int(h.best_epoch.iloc[0])
            selected = h[h.epoch.eq(best)]
            ax.scatter(selected.epoch, selected.validation_bce, marker='x', s=70)
        ax.set(xlabel='Эпоха (0 — до обновлений)', ylabel='Бинарная кросс-энтропия без регуляризации',
               title='Обучение матричной факторизации; крестик — выбранная эпоха')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        _save(fig, figures, '06_mf_learning')
        images.append(('06_mf_learning', 'Остановка и выбор модели — только по проверочной ошибке; тест не используется.'))
    if _bar(aggregate, 'catalog_coverage_at_k', max_k, methods, figures, '07_coverage',
            f'Охват каталога в рекомендациях, K={max_k}', 'Доля стратегий, попавших хотя бы в один список'):
        images.append(('07_coverage', 'Охват не является самостоятельной мерой качества рекомендаций.'))

    def value(method, metric, k):
        row = aggregate[(aggregate.method == method) & (aggregate.metric == metric) & (aggregate.k == k)]
        if row.empty or not np.isfinite(row['mean'].iloc[0]):
            return 'не определено'
        r = row.iloc[0]
        return f'{r["mean"]:.4f}' + (f' ± {r["std"]:.4f}' if r.n_seeds > 1 else '')

    notes = [
        'Это первые сравнения в синтетической среде, а не подтверждение финансовой полезности на реальной платформе.',
        'Все модели в пределах одного seed используют одни и те же обучающие, проверочные и тестовые истории. '
        'Разделение выполняется по глобальным шагам; один показанный список никогда не разрывается.',
        'Матричная факторизация обучается на идентификаторах и фактах выбора показанных стратегий. '
        'Истинная совместимость, архетипы и внутренние вероятности не являются её входами или целевыми метками.',
        'Публичная популярность — заданный начальный признак мира. Популярность по истории — число копирований только в обучающей части. '
        'Ориентир совместимости и исходная смешанная политика используют привилегированное знание симулятора.',
        'Оценка по полному каталогу измеряет соответствие заданной совместимости. Оценка на тестовых показах измеряет '
        'восстановление наблюдаемого выбора при старом механизме показа; она не является внеполитической оценкой.',
        'В тестовом ранжировании исключены списки без копирования; их количество сохранено в split_summary.json. '
        'Невыбранные показанные стратегии — наблюдения отсутствия выбора в данной ситуации, а не доказанная непригодность.',
        'Для новых показов используется замороженная модель и сброшенная копия того же мира. '
        'Сохраняются ожидаемая вероятность копирования и фактическая частота. Результаты не возвращаются в обучение.',
        'Погрешности на графиках — стандартное отклонение по независимым seed миров, не доверительные интервалы. '
        'В одном запуске разброс между мирами не оценивается. Неопределённость пользовательского поведения также остаётся.',
        'Историческая генерация шумна и смещена показами; простая матричная факторизация не обязана побеждать популярность. '
        'Фиксированные профили, финансовые прокси, повторные копирования и отсутствие позиционного эффекта сохранены из прототипа.',
    ]
    lines = ['# Сравнение рекомендательных политик и централизованной модели', '',
             f'Независимых миров: **{len(config["seeds"])}**; seed: {config["seeds"]}.', '',
             '## Сводка', '',
             f'| Метод | Совместимость@{max_k} | Разрыв до ориентира | NDCG@{choice_k} внутри тестового показа | Вероятность копирования |',
             '|---|---:|---:|---:|---:|']
    for method in methods:
        lines.append(f'| {LABELS[method]} | {value(method,"compatibility_at_k",max_k)} | '
                     f'{value(method,"oracle_regret_at_k",max_k)} | {value(method,"ndcg_at_k",choice_k)} | '
                     f'{value(method,"expected_copy_rate",0)} |')
    lines += ['', '## Как читать результаты', ''] + [f'{n+1}. {t}' for n,t in enumerate(notes)]
    lines += ['', '## Графики', '']
    for filename, caption in images:
        lines += [f'![{caption}](figures/{filename}.png)', '', caption, '']
    lines += ['## Протокол и источники', '',
              'Разделение мира, рекомендателя и отклика следует общей идее RecSim; это не перенос его кода. '
              'Использована собственная реализация логистической матричной факторизации на NumPy. '
              'Производные проверены численным дифференцированием.', '',
              '- Ie et al. RecSim: https://arxiv.org/abs/1909.04847',
              '- Krichene, Rendle. On Sampled Metrics for Item Recommendation: '
              'https://research.google/pubs/on-sampled-metrics-for-item-recommendation/',
              '- NumPy. Parallel random number generation: https://numpy.org/doc/stable/reference/random/parallel.html', '',
              'Финансовые формулы прототипа не изменены. Сбор вероятностей здесь не объявляется готовым протоколом '
              'внеполитической оценки: условная вероятность позиции не равна вероятности включения объекта в список.']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Self-contained local viewer except for relative PNGs shipped with the run.
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
            '<title>Результаты эксперимента</title>',
            '<style>body{font:16px/1.5 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid;padding:6px}'
            'img{max-width:100%;height:auto}h2{margin-top:32px}</style>',
            '<h1>Сравнение рекомендаций и матричной факторизации</h1>',
            f'<p>Миров: {len(config["seeds"])}; seed: {escape(str(config["seeds"]))}.</p>',
            '<table><tr><th>Метод</th><th>Совместимость</th><th>Разрыв</th><th>NDCG выбора</th><th>Копирование</th></tr>']
    for method in methods:
        cells = [LABELS[method], value(method,'compatibility_at_k',max_k),
                 value(method,'oracle_regret_at_k',max_k), value(method,'ndcg_at_k',choice_k),
                 value(method,'expected_copy_rate',0)]
        html.append('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'</tr>')
    html += ['</table>', '<h2>Условия интерпретации</h2>'] + [f'<p>{escape(t)}</p>' for t in notes]
    for filename, caption in images:
        html += [f'<h2>{escape(caption)}</h2>',
                 f'<img src="figures/{filename}.png" alt="{escape(caption)}">']
    html += ['<p>Полный протокол и источники: report.md; числовые данные: metrics_per_seed.csv и metrics_aggregate.csv.</p></html>']
    (root/'report.html').write_text('\n'.join(html), encoding='utf-8')
