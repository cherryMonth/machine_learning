# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field


class LianjiaItem(scrapy.Item):
    title = Field()  # 标题
    region = Field()  # 区域
    sub_region = Field()  # 子区域
    area = Field()  # 面积
    location = Field()  # 位置
    room = Field()  # 室
    hall = Field()  # 厅
    toilet = Field()  # 卫
    brand = Field()  # 品牌
    time = Field()  # 发布日期
    layer = Field()  # 层
    floor = Field()  # 楼层数
    price = Field()  # 价格
    tag = Field()  # 特色标签
    tenancy = Field()  # 租期
