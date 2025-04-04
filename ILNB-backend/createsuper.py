import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ilnb.settings")  # update if different
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

if not User.objects.filter(username="admin").exists():
    User.objects.create_superuser(
        username="iamnitishsah",
        email="3nitishkumar0@gmail.com",
        password="Nitish@2005"
    )
    print("Superuser created!")
else:
    print("Superuser already exists.")
