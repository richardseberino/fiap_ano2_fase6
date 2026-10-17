"""Treina, avalia e salva o modelo preditivo de pico de risco cardíaco.

Uso (a partir da raiz do repositório):
    python src/ml/treinar_modelo.py

Etapas, na mesma ordem do notebook:
1. Geração da base sintética (seed 42, 5.000 registros).
2. Separação treino/teste estratificada (80/20).
3. Comparação de três classificadores supervisionados com validação cruzada.
4. Avaliação do modelo escolhido no conjunto de teste.
5. Salvamento do modelo, das métricas e das figuras.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from ml.base_sintetica import FEATURES, TARGET, gerar_base  # noqa: E402

CAMINHO_MODELO = RAIZ / "src" / "models" / "modelo_pico_risco.joblib"
CAMINHO_METRICAS = RAIZ / "src" / "models" / "metricas_modelo.json"
CAMINHO_BASE = RAIZ / "data" / "base_sintetica_pico_risco.csv"
PASTA_IMG = RAIZ / "docs" / "img"

SEED = 42


def candidatos() -> dict:
    """Classificadores supervisionados comparados na escolha do algoritmo."""
    return {
        "Regressão Logística": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
    }


def avaliar(modelo, X_teste, y_teste) -> dict:
    """Calcula as métricas de classificação no conjunto de teste."""
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    vn, fp, fn, vp = confusion_matrix(y_teste, y_pred).ravel()
    return {
        "acuracia": round(accuracy_score(y_teste, y_pred), 4),
        "precisao": round(precision_score(y_teste, y_pred), 4),
        "recall": round(recall_score(y_teste, y_pred), 4),
        "f1": round(f1_score(y_teste, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_teste, y_prob), 4),
        "matriz_confusao": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
    }


def treinar(salvar_figuras: bool = True, verbose: bool = True) -> dict:
    """Executa o pipeline completo e devolve o pacote salvo do modelo."""
    df = gerar_base()
    CAMINHO_BASE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_BASE, index=False)

    X, y = df[FEATURES], df[TARGET]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Comparação dos candidatos com validação cruzada estratificada (5 folds) no treino.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparacao = {}
    for nome, modelo in candidatos().items():
        f1_cv = cross_val_score(modelo, X_treino, y_treino, cv=cv, scoring="f1")
        modelo.fit(X_treino, y_treino)
        comparacao[nome] = {
            "f1_cv_media": round(f1_cv.mean(), 4),
            "f1_cv_desvio": round(f1_cv.std(), 4),
            **avaliar(modelo, X_teste, y_teste),
        }
        if verbose:
            print(f"{nome:<22} F1 (CV) = {f1_cv.mean():.4f} ± {f1_cv.std():.4f}")

    escolhido = "Random Forest"
    modelo = candidatos()[escolhido].fit(X_treino, y_treino)
    metricas = avaliar(modelo, X_teste, y_teste)

    importancias = (
        pd.Series(modelo.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
        .round(4)
    )

    if verbose:
        print(f"\nModelo escolhido: {escolhido}")
        print(classification_report(y_teste, modelo.predict(X_teste), digits=4))
        print("Matriz de confusão:", metricas["matriz_confusao"])

    if salvar_figuras:
        PASTA_IMG.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_estimator(
            modelo, X_teste, y_teste,
            display_labels=["Sem pico (0)", "Pico (1)"], cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title("Matriz de confusão: Random Forest (teste)")
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        fig.tight_layout()
        fig.savefig(PASTA_IMG / "matriz_confusao.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.6))
        importancias.sort_values().plot.barh(ax=ax, color="#2563eb")
        ax.set_title("Importância das variáveis: Random Forest")
        ax.set_xlabel("Importância (redução média de impureza)")
        fig.tight_layout()
        fig.savefig(PASTA_IMG / "importancia_variaveis.png", dpi=150)
        plt.close(fig)

    pacote = {
        "modelo": modelo,
        "algoritmo": escolhido,
        "features": FEATURES,
        "target": TARGET,
        "metricas_teste": metricas,
        "versao_sklearn": sklearn.__version__,
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
    }
    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, CAMINHO_MODELO)

    relatorio = {
        "algoritmo_escolhido": escolhido,
        "registros": len(df),
        "proporcao_pico_risco": round(float(y.mean()), 4),
        "tamanho_treino": len(X_treino),
        "tamanho_teste": len(X_teste),
        "comparacao_modelos": comparacao,
        "metricas_teste": metricas,
        "importancia_variaveis": importancias.to_dict(),
        "versao_sklearn": sklearn.__version__,
    }
    CAMINHO_METRICAS.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\nModelo salvo em: {CAMINHO_MODELO.relative_to(RAIZ)}")
        print(f"Métricas salvas em: {CAMINHO_METRICAS.relative_to(RAIZ)}")
    return pacote


if __name__ == "__main__":
    treinar()
