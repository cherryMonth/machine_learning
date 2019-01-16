# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field


class DoubanItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = Field()  # 名字
    douban_id = Field()
    director = Field()  # 导演
    screenwriter = Field()  # 编剧
    tag = Field()  # 类型
    region = Field()  # 制作地区
    language = Field()  # 语言
    release_time = Field()  # 上映时间
    film_length = Field()  # 片长
    star = Field()  # 主演
    collections = Field()  # 评价人数
    rating_num = Field()  # 评分
    five_star = Field()
    four_star = Field()
    three_star = Field()
    two_star = Field()
    one_star = Field()
    intro = Field()
