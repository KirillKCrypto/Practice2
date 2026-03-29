from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
from scipy import stats

DATA_FILE = Path(__file__).resolve().parent / 'UN_Tourism_8_9_1_TDGDP_04_2025.xlsx'
OUT_DIR = Path(__file__).resolve().parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

# Official SIDS-style group used for independent comparison in the report.
SIDS = {
    'Antigua and Barbuda', 'Bahamas', 'Bahrain', 'Barbados', 'Belize', 'Cabo Verde',
    'Comoros', 'Cuba', 'Dominica', 'Dominican Republic', 'Fiji', 'Grenada', 'Guinea-Bissau',
    'Guyana', 'Haiti', 'Jamaica', 'Kiribati', 'Maldives', 'Marshall Islands', 'Mauritius',
    'Micronesia', 'Nauru', 'Palau', 'Papua New Guinea', 'Saint Kitts and Nevis',
    'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'Sao Tome and Principe',
    'Seychelles', 'Singapore', 'Solomon Islands', 'Suriname', 'Timor-Leste', 'Tonga',
    'Trinidad and Tobago', 'Tuvalu', 'Vanuatu'
}


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name='SDG 8.9.1')
    df = df[['GeoAreaCode', 'GeoAreaName', 'TimePeriod', 'Value']].dropna(subset=['Value']).copy()
    df['GeoAreaCode'] = df['GeoAreaCode'].astype(int)
    df['TimePeriod'] = df['TimePeriod'].astype(int)
    df['Value'] = df['Value'].astype(float)
    return df


def year_sample(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return (
        df[df['TimePeriod'] == year]
        .sort_values(['GeoAreaName', 'GeoAreaCode'])
        .reset_index(drop=True)
    )


def paired_sample(df: pd.DataFrame, year_a: int, year_b: int) -> pd.DataFrame:
    a = year_sample(df, year_a)[['GeoAreaCode', 'GeoAreaName', 'Value']].rename(columns={'Value': f'Value_{year_a}'})
    b = year_sample(df, year_b)[['GeoAreaCode', 'GeoAreaName', 'Value']].rename(columns={'Value': f'Value_{year_b}'})
    return a.merge(b, on=['GeoAreaCode', 'GeoAreaName'], how='inner').sort_values('GeoAreaName').reset_index(drop=True)


def describe_series(s: pd.Series) -> dict[str, float]:
    return {
        'n': int(s.shape[0]),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=1)),
        'min': float(s.min()),
        'q1': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q3': float(s.quantile(0.75)),
        'max': float(s.max()),
    }


def fmt(x: float) -> str:
    return f'{x:.3f}'


def main() -> None:
    df = load_data()
    y2019 = year_sample(df, 2019)
    y2021 = year_sample(df, 2021)
    pair = paired_sample(df, 2019, 2021)

    shapiro = stats.shapiro(y2019['Value'])
    spearman = stats.spearmanr(pair['Value_2019'], pair['Value_2021'], alternative='two-sided')
    kendall = stats.kendalltau(pair['Value_2019'], pair['Value_2021'], alternative='two-sided', nan_policy='omit')
    wilcoxon = stats.wilcoxon(
        pair['Value_2019'], pair['Value_2021'],
        zero_method='wilcox', correction=False,
        alternative='two-sided', method='auto'
    )

    y2019_groups = y2019.copy()
    y2019_groups['Group'] = y2019_groups['GeoAreaName'].apply(lambda x: 'SIDS' if x in SIDS else 'Non-SIDS')
    sids = y2019_groups[y2019_groups['Group'] == 'SIDS']['Value']
    non_sids = y2019_groups[y2019_groups['Group'] == 'Non-SIDS']['Value']
    mann_whitney = stats.mannwhitneyu(sids, non_sids, alternative='two-sided', method='auto', use_continuity=True)

    report = {
        'descriptive_2019': describe_series(y2019['Value']),
        'descriptive_2021': describe_series(y2021['Value']),
        'paired_n_2019_2021': int(pair.shape[0]),
        'sids_n_2019': int(sids.shape[0]),
        'non_sids_n_2019': int(non_sids.shape[0]),
        'tests': {
            'shapiro_2019': {'W': float(shapiro.statistic), 'p_value': float(shapiro.pvalue)},
            'spearman_2019_2021': {'rho': float(spearman.statistic), 'p_value': float(spearman.pvalue)},
            'kendall_2019_2021': {'tau': float(kendall.statistic), 'p_value': float(kendall.pvalue)},
            'wilcoxon_2019_2021': {'T': float(wilcoxon.statistic), 'p_value': float(wilcoxon.pvalue)},
            'mannwhitney_sids_non_sids_2019': {'U': float(mann_whitney.statistic), 'p_value': float(mann_whitney.pvalue)},
        },
        'largest_declines_2019_2021': (
            pair.assign(delta=pair['Value_2021'] - pair['Value_2019'])
            .sort_values('delta')
            .head(10)
            [['GeoAreaName', 'Value_2019', 'Value_2021', 'delta']]
            .to_dict(orient='records')
        ),
    }

    with open(OUT_DIR / 'statistics_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary_lines = [
        'Практическая работа № 2 — статистические результаты',
        '',
        f"2019: n={report['descriptive_2019']['n']}, mean={fmt(report['descriptive_2019']['mean'])}, median={fmt(report['descriptive_2019']['median'])}",
        f"2021: n={report['descriptive_2021']['n']}, mean={fmt(report['descriptive_2021']['mean'])}, median={fmt(report['descriptive_2021']['median'])}",
        f"Парная выборка 2019/2021: n={report['paired_n_2019_2021']}",
        f"SIDS / Non-SIDS (2019): {report['sids_n_2019']} / {report['non_sids_n_2019']}",
        '',
        f"Shapiro-Wilk (2019): W={fmt(shapiro.statistic)}, p={shapiro.pvalue:.6g}",
        f"Spearman (2019 vs 2021): rho={fmt(spearman.statistic)}, p={spearman.pvalue:.6g}",
        f"Kendall (2019 vs 2021): tau={fmt(kendall.statistic)}, p={kendall.pvalue:.6g}",
        f"Wilcoxon (2019 vs 2021): T={fmt(wilcoxon.statistic)}, p={wilcoxon.pvalue:.6g}",
        f"Mann-Whitney (SIDS vs Non-SIDS, 2019): U={fmt(mann_whitney.statistic)}, p={mann_whitney.pvalue:.6g}",
    ]
    (OUT_DIR / 'statistics_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    print('\n'.join(summary_lines))
    print(f'\nFiles saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
