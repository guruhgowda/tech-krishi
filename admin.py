from django.contrib import admin
from .models import userdetails,cropdetails,pestdetect,weatherreport,rentingtool,expertadvice

# Register your models here.

admin.site.register(userdetails)
admin.site.register(cropdetails)
admin.site.register(pestdetect)
admin.site.register(weatherreport)
admin.site.register(rentingtool)
admin.site.register(expertadvice)
