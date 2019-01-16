# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field


class ErshoucheItem(scrapy.Item):
    # define the fields for your item here like:
    brand = Field()  # 品牌
    title = Field()  # 标题
    price = Field()  # 价格(万元)
    status = Field()  # 是否急需出售
    start_time = Field()  # 汽车购买的起始时间
    distance = Field()  # 汽车跑过的总里程
    volumn = Field()  # 汽油容量　
    gear = Field()  # 档位（手动，自动）
    tag = Field()  # 商品的标签(售主会员年数、好评率、配置级别，是否有行驶证等)
    authentication = Field()  # 58是否认证
