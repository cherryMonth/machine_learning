# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field


class HousePriceItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    time = Field()
    city = Field()
    residence_MOM = Field()  # 环比
    residence_YOY = Field()  # 同比
    residence_FBI = Field()  # 定基指数
    commercial_residence_MOM = Field()  # 商品住宅 环比
    commercial_residence_YOY = Field()  # 同比
    commercial_residence_FBI = Field()  # 定基指数
    second_hand_MOM = Field()  # 二手住宅　环比
    second_hand_YOY = Field()  # 二手住宅　同比
    second_hand_FBI = Field()  # 二手住宅　定基指数
