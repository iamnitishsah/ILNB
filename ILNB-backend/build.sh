#!/usr/bin/env bash
# Install dependencies
pip install -r requirements.txt

# Run migrations and collect static
python manage.py migrate
python manage.py collectstatic --noinput
