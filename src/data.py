"""Módulo oficial de carregamento e sanidade dos dados (Tópico 1 -> Tópico 3)."""
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd

# Semente global de reprodutibilidade da equipe
RANDOM_SEED: int = 42

# URL oficial da base Diabetes 130-US hospitals (UCI Machine Learning Repository)
UCI_DIABETES_URL: str = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# IDs de alta correspondentes a óbito ou cuidados paliativos terminais (hospice)
# 11: Expired, 13: Hospice / home, 14: Hospice / medical facility,
# 19: Expired at home, 20: Expired in a medical facility, 21: Expired, place unknown
EXPIRED_DISCHARGE_IDS: list[int] = [11, 13, 14, 19, 20, 21]

# Definição canônica dos grupos estatísticos de variáveis (Contrato Tópico 1 -> Tópico 3)
NUMERIC_COUNT_COLS: list[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

ORDINAL_COLS: list[str] = [
    "age_years",
]

NOMINAL_CATEGORICAL_COLS: list[str] = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
]

MEDICATION_COLS: list[str] = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]


def ensure_raw_data(data_dir: str | Path = "data") -> Path:
    """
    Garante que o arquivo bruto `diabetic_data.csv` esteja presente em `data_dir`.
    Caso não exista, realiza o download automático do repositório oficial da UCI.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / "diabetic_data.csv"

    if not csv_path.exists():
        zip_path = data_path / "diabetes_uci.zip"
        print(f"[data.py] Baixando base oficial da UCI em {zip_path}...")
        urllib.request.urlretrieve(UCI_DIABETES_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
        print(f"[data.py] Base extraída com sucesso em {csv_path}.")

    return csv_path


def map_icd9_to_category(code: object) -> str:
    """
    Mapeia códigos CID-9 (ICD-9) para os 9 capítulos fisiopatológicos
    definidos por Strack et al. (2014):
    Circulatory, Respiratory, Digestive, Diabetes, Injury,
    Musculoskeletal, Genitourinary, Neoplasms, Other (e Missing).
    """
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if not s or s == "?":
        return "Missing"

    # Códigos suplementares V (fatores de saúde) ou E (causas externas)
    if s.startswith(("V", "E")):
        return "Other"

    # Código 250.xx corresponde diretamente a Diabetes Mellitus
    if s.startswith("250"):
        return "Diabetes"

    try:
        val = float(s)
    except ValueError:
        return "Other"

    if (390 <= val <= 459) or val == 785:
        return "Circulatory"
    if (460 <= val <= 519) or val == 786:
        return "Respiratory"
    if (520 <= val <= 579) or val == 787:
        return "Digestive"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if (580 <= val <= 629) or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def load_raw_diabetes(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Carrega o CSV bruto preservando a categoria clínica 'None' (exame não solicitado)
    nas variáveis laboratoriais (`A1Cresult` e `max_glu_serum`) e tratando apenas '?' como NaN.
    """
    csv_path = ensure_raw_data(data_dir)
    return pd.read_csv(
        csv_path,
        keep_default_na=False,
        na_values=["?"],
        low_memory=False,
    )


def load_sanitized_data(
    data_dir: str | Path = "data",
    target_mode: str = "any_readmission",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Executa todas as regras de Checagem e Sanidade dos Dados (Seção 1):
    1. Remove registros com desfecho de óbito ou hospice (evita Data Leakage).
    2. Remove registros com gênero inválido ('Unknown/Invalid').
    3. Garante independência amostral (i.i.d.) mantendo apenas a 1ª internação por `patient_nbr`.
    4. Agrupa os códigos CID-9 (`diag_1`, `diag_2`, `diag_3`) em 9 categorias clínicas.
    5. Converte a faixa etária ordinal `age` para ponto médio numérico (`age_years`: 5 a 95).
    6. Descarta identificadores (`encounter_id`, `patient_nbr`), atributos com ausência
       extrema/administrativos (`weight`, `payer_code`, `medical_specialty`) e colunas
       constantes de variância zero (`examide`, `citoglipton`).

    Parâmetros
    ----------
    data_dir : str | Path
        Diretório onde reside `diabetic_data.csv`.
    target_mode : str
        - "any_readmission" (padrão): y = 1 se readmitido (<30 ou >30), y = 0 se 'NO'.
        - "early_30d": y = 1 se readmitido em <30 dias, y = 0 caso contrário.

    Retorna
    -------
    X : pd.DataFrame (N = 69.987 linhas, d = 42 colunas tipadas)
    y : pd.Series (vetor binário 0/1)
    """
    df = load_raw_diabetes(data_dir)

    # 1. Remoção de óbitos/hospice (Data Leakage) e anomalias de cadastro
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)]
    df = df[df["gender"] != "Unknown/Invalid"]

    # 2. Independência amostral: manter apenas o 1º encontro de cada paciente único
    df = (
        df.sort_values("encounter_id")
        .drop_duplicates(subset=["patient_nbr"], keep="first")
        .reset_index(drop=True)
    )

    # 3. Definição da variável alvo binária y
    if target_mode == "any_readmission":
        y = (df["readmitted"] != "NO").astype(int).rename("target_readmitted")
    elif target_mode == "early_30d":
        y = (df["readmitted"] == "<30").astype(int).rename("target_early_30d")
    else:
        raise ValueError(f"target_mode inválido: '{target_mode}'. Use 'any_readmission' ou 'early_30d'.")

    # 4. Agrupamento clínico CID-9 das variáveis de diagnóstico
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        df[diag_col] = df[diag_col].apply(map_icd9_to_category)

    # 5. Mapeamento ordinal da faixa etária para ponto médio em anos
    age_map = {f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)}
    df["age_years"] = df["age"].map(age_map).astype(float)

    # 6. Conversão explícita de IDs administrativos para string nominal
    for id_col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[id_col] = "ID_" + df[id_col].astype(str)

    # 7. Seleção ordenada dos 42 atributos preditivos validados
    ordered_feature_cols = (
        NUMERIC_COUNT_COLS
        + ORDINAL_COLS
        + NOMINAL_CATEGORICAL_COLS
        + MEDICATION_COLS
    )
    X = df[ordered_feature_cols].copy()

    return X, y


def get_feature_groups() -> dict[str, list[str]]:
    """
    Retorna o dicionário com a separação de colunas por tipo estatístico
    para uso direto no `ColumnTransformer` da Seção 3.
    """
    return {
        "numeric_count": NUMERIC_COUNT_COLS.copy(),
        "ordinal": ORDINAL_COLS.copy(),
        "nominal_categorical": NOMINAL_CATEGORICAL_COLS.copy(),
        "medications": MEDICATION_COLS.copy(),
    }


if __name__ == "__main__":
    X_clean, y_clean = load_sanitized_data()
    grupos = get_feature_groups()
    print("=" * 72)
    print("TESTE DO MÓDULO src/data.py — DADOS SANITIZADOS PRONTOS PARA A SEÇÃO 3")
    print("=" * 72)
    print(f"• Matriz X : {X_clean.shape[0]:,} pacientes únicos × {X_clean.shape[1]} atributos")
    print(f"• Vetor y  : {y_clean.mean():.2%} positivos ({y_clean.sum():,} readmitidos)")
    print(f"• Grupos   : { {k: len(v) for k, v in grupos.items()} }")
    print("=" * 72)
