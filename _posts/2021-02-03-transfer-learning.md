---
layout: post
title: Transfer learning
subtitle: یادگیری انتقالی
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [یادگیری انتقالی, transfer learning]
---

با استفاده از یادگیری انتقالی 
(transfer learning) 
می توان از مدل هایی که قبلا روی دیتاست ها برزگ آموزش دیده اند استفاده کرده و یک شبکه ی عصبی پیاده سازی کرد وزن هایی که این مدل ها بعد از 
آموزش بدست اوردن موجود بوده و در فریمورک های مختلف می توان از ان استفاده کرد در ادامه مثالی از یادگیری انتقالی با استفاده از فریمورک کراس و مدل 
VGG16
رو بررسی می کنم 
در ابتدا کتابخانه های مورد نیاز در فریمورک کراس را ایمپورت میکنیم 

.. TEASER_END

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

با استفاده از دستور زیر مدل را تعریف کرده و از وزن های بدست آمده روی دیتاست 
imagenet
برای مدل 
VGG16
استفاده میکنیم 


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

با استفاده از دستور زیر مشخص میکنیم که مدل نیاز به اموزش ندارد و از وزن های بدست امده در قسمت 
VGG16
استفاده شود 

vgg.trainable = False

با استفاده از دستور زیر مدل  ترتیبی تعریف کرده و پارامتر ها و وزن های مدل 
VGG16 
را به ان اضافه میکنیم  و در ادامه لایه های مورد درنظر برای دیتاست را اضافه کرده تا بتوانیم مسله مورد نظر را حل کنیم
در این مثال از این مدل برای تشخیص جمسیت استفاده شده و مدل برای جنسیت تصاویر ورودی به دو کلاس مرد و زن تعریف شده است

 
model = Sequential()

model.add(vgg)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


مدل را کامپایل کرده و تابع هزینه مشخص شده و فرایند آموزش روی دادهها انجام می شود 
