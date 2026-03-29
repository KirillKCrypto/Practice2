from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).resolve().parent / 'UN_Tourism_8_9_1_TDGDP_04_2025.xlsx'
OUT_DIR = Path(__file__).resolve().parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

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
    return df[df['TimePeriod'] == year].sort_values(['GeoAreaName', 'GeoAreaCode']).reset_index(drop=True)


def paired_sample(df: pd.DataFrame, year_a: int, year_b: int) -> pd.DataFrame:
    a = year_sample(df, year_a)[['GeoAreaCode', 'GeoAreaName', 'Value']].rename(columns={'Value': f'Value_{year_a}'})
    b = year_sample(df, year_b)[['GeoAreaCode', 'GeoAreaName', 'Value']].rename(columns={'Value': f'Value_{year_b}'})
    return a.merge(b, on=['GeoAreaCode', 'GeoAreaName'], how='inner').sort_values('GeoAreaName').reset_index(drop=True)


def save_histogram(y2019: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5.5))
    plt.hist(y2019['Value'], bins=15, edgecolor='black')
    plt.xlabel('Tourism direct GDP share, %')
    plt.ylabel('Number of countries')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure_1_histogram_2019.png', dpi=300)
    plt.close()


def save_top10(y2019: pd.DataFrame) -> None:
    top10 = y2019.nlargest(10, 'Value').sort_values('Value')
    plt.figure(figsize=(8.5, 5.8))
    plt.barh(top10['GeoAreaName'], top10['Value'], edgecolor='black')
    plt.xlabel('Tourism direct GDP share, %')
    plt.ylabel('Country / territory')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure_2_top10_2019.png', dpi=300)
    plt.close()


def save_scatter(pair: pd.DataFrame) -> None:
    x = pair['Value_2019']
    y = pair['Value_2021']
    limit = max(x.max(), y.max()) + 1
    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(x, y, alpha=0.75, edgecolors='black')
    plt.plot([0, limit], [0, limit], linestyle='--', label='y = x')
    plt.xlabel('2019, %')
    plt.ylabel('2021, %')
    plt.legend()
    plt.grid(linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure_3_scatter_2019_2021.png', dpi=300)
    plt.close()


def save_boxplot(y2019: pd.DataFrame) -> None:
    y2019 = y2019.copy()
    y2019['Group'] = y2019['GeoAreaName'].apply(lambda x: 'SIDS' if x in SIDS else 'Non-SIDS')
    data = [y2019[y2019['Group'] == 'SIDS']['Value'], y2019[y2019['Group'] == 'Non-SIDS']['Value']]
    plt.figure(figsize=(7.2, 5.8))
    plt.boxplot(data, tick_labels=['SIDS', 'Non-SIDS'])
    plt.ylabel('Tourism direct GDP share, %')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure_4_boxplot_sids.png', dpi=300)
    plt.close()


def main() -> None:
    df = load_data()
    y2019 = year_sample(df, 2019)
    pair = paired_sample(df, 2019, 2021)
    save_histogram(y2019)
    save_top10(y2019)
    save_scatter(pair)
    save_boxplot(y2019)
    print(f'Figures saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
