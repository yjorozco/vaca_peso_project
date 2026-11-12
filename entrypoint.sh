#!/bin/sh
set -e

echo "==> [1/3] Generando migraciones de modelos..."
python manage.py makemigrations

echo "==> [2/3] Aplicando migraciones a SQLite..."
python manage.py migrate

echo "==> [3/3] Iniciando servidor de desarrollo Django en 0.0.0.0:8000..."
exec python manage.py runserver 0.0.0.0:8000