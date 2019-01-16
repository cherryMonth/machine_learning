# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float

# 导入会话构建对象
from sqlalchemy.orm import sessionmaker
import pandas as pd
from scrapy import log

log.msg("This is a warning", level=log.WARNING)


class DoubanPipeline(object):
    def __init__(self):
        '''
        初始化对象数据：可以用于初始化资源
            如：打开文件，打开数据库连接等操作
        '''
        # 创建连接数据库引擎
        self.engine = create_engine('sqlite:///douban.sqlite3', echo=True)
        metadata = MetaData(self.engine)
        item = Table('item', metadata, Column('id', Integer, primary_key=True),
                     Column('douban_id', Integer, unique=True),
                     Column('name', String(50), unique=True),
                     Column('director', String(50)),
                     Column('screenwriter', String(50)),
                     Column('tag', String(50)),
                     Column('region', String(50)),
                     Column('language', String(50)),
                     Column('release_time', String(50)),
                     Column('film_length', String(50)),
                     Column('star', String(300)),
                     Column('collections', Integer),
                     Column('rating_num', Float),
                     Column('five_star', Integer),
                     Column('four_star', Integer),
                     Column('three_star', Integer),
                     Column('two_star', Integer),
                     Column('one_star', Integer),
                     Column('intro', String(1000))
                     )
        metadata.create_all(self.engine)
        session = sessionmaker(bind=self.engine)
        self.session = session()

    def process_item(self, item, spider):
        '''
        该函数会在爬虫采集并封装好的Item对象时自动调用
        函数针对item数据进行验证和存储
        :param item:
        :param spider:
        :return:
        '''

        # 定义sql语句
        sql = 'insert into item(name, douban_id, director, screenwriter, tag, region, language, release_time, film_length, star,' \
              'collections, five_star, four_star, three_star, two_star, one_star, intro, rating_num)' \
              ' values ( "%s", "%s", "%s", ' \
              '"%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' \
              % (item['name'], item['douban_id'], item['director'], item['screenwriter'], item['tag'], item['region'],
                 item['language'], item['release_time'], item['film_length'], item['star'], item['collections'],
                 item['five_star'], item['four_star'], item['three_star'], item['two_star'], item['one_star'],
                 item['intro'], item['rating_num'])
        # 执行sql语句
        self.session.execute(sql)
        # 提交数据
        self.session.commit()

    def close_spider(self, spider):
        '''
        爬虫程序自动关闭时调用函数
        经常用于做一些资源回收工作，如关闭和数据库的连接
        :return:
        '''
        data = pd.read_sql("select * from item;", con=self.engine)
        data.to_csv('douban.csv', index=False)
        self.session.close()

    def open_spider(self, spider):
        '''
        爬虫开启时需要调用函数，经常用于数据的初始化
        :param spider:
        :return:
        '''
        pass
