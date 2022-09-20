from django.db import models

# Create your models here.
class Person(models.Model):
    image = models.ImageField(default=None)
    identity_number = models.IntegerField(default=0)
