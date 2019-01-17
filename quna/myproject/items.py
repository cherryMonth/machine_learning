# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field

class MyprojectItem(scrapy.Item):
    # define the fields for your item here like:
    name = Field()
    level = Field()
    hot = Field()
    area = Field()
    price = Field()
    info = Field()
    address = Field()
    num = Field()

