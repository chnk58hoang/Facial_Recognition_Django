# Generated by Django 3.2.5 on 2022-08-08 04:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('facenet512', '0002_remove_person_identity_number'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='identity_number',
            field=models.IntegerField(default=0),
        ),
    ]