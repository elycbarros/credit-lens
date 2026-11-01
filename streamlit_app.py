"""Ponto de entrada para execução do dashboard no Streamlit Community Cloud e localmente.

Uso:
    streamlit run streamlit_app.py
    # ou via make:
    make dashboard
"""

import sys
from pathlib import Path

# Garante que 'src' esteja acessível no PYTHONPATH
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from credit_lens.dashboard import main  # noqa: E402

if __name__ == "__main__":
    main()
