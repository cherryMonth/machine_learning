# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float

# 导入会话构建对象
from sqlalchemy.orm import sessionmaker
import pandas as pd


class ErshouchePipeline(object):
    def __init__(self):
        '''
        初始化对象数据：可以用于初始化资源
            如：打开文件，打开数据库连接等操作
        '''
        # 创建连接数据库引擎
        self.engine = create_engine('sqlite:///ershouche.sqlite3', echo=True)
        metadata = MetaData(self.engine)
        item = Table('item', metadata, Column('id', Integer, primary_key=True),
                     Column('brand', String(50)),
                     Column('title', String(100)),
                     Column('price', Float),
                     Column('status', String(20)),
                     Column('start_time', String(20)),
                     Column('distance', String(20)),
                     Column('volumn', String(20)),
                     Column('gear', String(10)),
                     Column('tag', String(50))
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
        sql = 'insert into item(brand, title, price, status, start_time, distance, volumn, gear, tag) values ' \
              '( "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' \
              % (item['brand'], item['title'], item['price'], item['status'], item['start_time'], item['distance'],
                 item['volumn'], item['gear'], item['tag'])
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
        data.to_csv('ershouche.csv', index=False)
        self.session.close()

    def open_spider(self, spider):
        '''
        爬虫开启时需要调用函数，经常用于数据的初始化
        :param spider:
        :return:
        '''
        pass
