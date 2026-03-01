#!/bin/sh
set -eu

CERT_DIR="/etc/nginx/certs"
CERT_FILE="${CERT_DIR}/sherpa-gateway.crt"
KEY_FILE="${CERT_DIR}/sherpa-gateway.key"
TLS_HOST="${SHERPA_GATEWAY_TLS_HOST:-sherpa-gateway.sherpa.orb.local}"
TLS_DAYS="${SHERPA_GATEWAY_TLS_DAYS:-3650}"

mkdir -p "${CERT_DIR}"

if [ ! -s "${CERT_FILE}" ] || [ ! -s "${KEY_FILE}" ]; then
  echo "[gateway] generating self-signed TLS cert for ${TLS_HOST}"
  openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -days "${TLS_DAYS}" \
    -subj "/CN=${TLS_HOST}" \
    -addext "subjectAltName=DNS:${TLS_HOST},DNS:localhost,IP:127.0.0.1"
fi

exec nginx -g "daemon off;"
