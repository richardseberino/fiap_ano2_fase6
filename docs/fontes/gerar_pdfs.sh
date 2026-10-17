#!/usr/bin/env bash
# Gera os PDFs da pasta docs a partir dos HTML desta pasta usando o Google Chrome em modo headless.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
for nome in relatorio_tecnico arquitetura_multiagente; do
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="../$nome.pdf" "file://$PWD/$nome.html" 2>/dev/null
  echo "Gerado: docs/$nome.pdf"
done
