from django.db import models


# Create your models here.


class userdetails(models.Model):
    id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=30, null=False)
    last_name = models.CharField(max_length=30)
    emailid = models.CharField(max_length=30, null=False)
    password = models.CharField(max_length=30, null=False)
    phonenumber = models.CharField(max_length=30, null=False)

    def __str__(self):
        return "%s %s %s %s %s %s" % (
            self.id, self.first_name, self.last_name, self.emailid, self.password, self.phonenumber)


class cropdetails(models.Model):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=30, null=False)
    nitrogen = models.CharField(max_length=30)
    phosphorous = models.CharField(max_length=30, null=False)
    potassium = models.CharField(max_length=30, null=False)
    crop_predict = models.CharField(max_length=30, null=False)
    fertilize_predict = models.CharField(max_length=30, null=False)

    def __str__(self):
        return "%s %s %s %s %s %s %s" % (
            self.id, self.userid, self.nitrogen, self.phosphorous, self.potassium, self.crop_predict,self.fertilize_predict)


class pestdetect(models.Model):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=30, null=False)
    imagename = models.CharField(max_length=50, null=False)
    pestname = models.CharField(max_length=50, null=False)

    def __str__(self):
        return self.id


class weatherreport(models.Model):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=30, null=False)
    locationname = models.CharField(max_length=50, null=False)
    temp = models.CharField(max_length=50, null=False)
    todaydate = models.CharField(max_length=50)


    def __str__(self):
        return "%s %s %s %s %s" % (self.id, self.userid, self.locationname, self.temp, self.todaydate )


class rentingtool(models.Model):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=30, null=False)
    toolname = models.CharField(max_length=50, null=False)
    toolcost = models.CharField(max_length=50, null=False)
    todaydate = models.CharField(max_length=50)

    def __str__(self):
        return "%s %s %s %s %s" % (self.id, self.userid, self.toolname, self.toolcost, self.todaydate)


class expertadvice(models.Model):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=30, null=False)
    Questionname = models.CharField(max_length=50, null=False)
    advices = models.CharField(max_length=50, null=False)
    todaydate = models.CharField(max_length=50)

    def __str__(self):
        return self.id
