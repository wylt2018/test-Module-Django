from django.db import models

# Create your models here.


class Sentence(models.Model):
    input = models.CharField()

    def __str__(self):
        return self.input
