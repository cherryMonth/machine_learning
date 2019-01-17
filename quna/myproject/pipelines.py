# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# 导入数据库引擎对象
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float

# 导入会话构建对象
from sqlalchemy.orm import sessionmaker
import pandas as pd


class HotAddressPipeline(object):

    def __init__(self):
        '''
        初始化对象数据：可以用于初始化资源
            如：打开文件，打开数据库连接等操作
        '''
        # 创建连接数据库引擎
        self.engine = create_engine('sqlite:///MyDB.sqlite3', echo=True)
        metadata = MetaData(self.engine)
        item = Table('item', metadata, Column('id', Integer, primary_key=True),
                     Column('name', String(30), unique=True), Column('level', String(20)), Column('hot', Float),
                     Column('area', String(50)), Column('num', Integer), Column('price', Float),
                     Column('info', String(200)), Column('address', String(50)))
        metadata.create_all(self.engine)
        session = sessionmaker(bind=self.engine)
        self.session = session()

    def close_spider(self, spider):
        '''
        爬虫程序自动关闭时调用函数
        经常用于做一些资源回收工作，如关闭和数据库的连接
        :return:
        '''
        data = pd.read_sql("select * from item;", con=self.engine)
        data.to_csv('hot.csv', index=False)
        self.session.close()

    def open_spider(self, spider):
        '''
        爬虫开启时需要调用函数，经常用于数据的初始化
        :param spider:
        :return:
        '''
        pass

    def process_item(self, item, spider):
        '''
        该函数会在爬虫采集并封装好的Item对象时自动调用
        函数针对item数据进行验证和存储
        :param item:
        :param spider:
        :return:
        '''

        # 定义sql语句
        sql = 'insert into item(name, level, hot, address, area, price, info, num) values ( "%s", "%s", ' \
              '"%s", "%s", "%s", "%s", "%s", "%s")' \
              % (item['name'], item['level'], item['hot'], item['address'], item['area'], item['price'], item['info'],
                 item['num'])
        # 执行sql语句
        self.session.execute(sql)
        # 提交数据
        self.session.commit()
