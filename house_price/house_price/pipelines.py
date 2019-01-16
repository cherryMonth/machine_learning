# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float

# 导入会话构建对象
from sqlalchemy.orm import sessionmaker
import pandas as pd


class HousePricePipeline(object):
    def __init__(self):
        '''
        初始化对象数据：可以用于初始化资源
            如：打开文件，打开数据库连接等操作
        '''
        # 创建连接数据库引擎
        self.engine = create_engine('sqlite:///house_price.sqlite3', echo=True)
        metadata = MetaData(self.engine)
        item = Table('item', metadata, Column('id', Integer, primary_key=True),
                     Column('time', String(20)),
                     Column('city', String(20)),
                     Column('residence_MOM', Float),
                     Column('residence_YOY', Float),
                     Column('residence_FBI', Float),
                     Column('commercial_residence_MOM', Float),
                     Column('commercial_residence_YOY', Float),
                     Column('commercial_residence_FBI', Float),
                     Column('second_hand_MOM', Float),
                     Column('second_hand_YOY', Float),
                     Column('second_hand_FBI', Float)
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
        sql = 'insert into item(time, city, residence_MOM, residence_YOY, residence_FBI, commercial_residence_MOM, ' \
              'commercial_residence_YOY, commercial_residence_FBI, second_hand_MOM, second_hand_YOY, second_hand_FBI) '\
              'values ( "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' \
              % (item['time'], item['city'], item['residence_MOM'], item['residence_YOY'], item['residence_FBI'],
                 item['commercial_residence_MOM'], item['residence_YOY'], item['residence_FBI'],
                 item['second_hand_MOM'], item['second_hand_YOY'], item['second_hand_FBI'])
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
        data.to_csv('house_price.csv', index=False)
        self.session.close()

    def open_spider(self, spider):
        '''
        爬虫开启时需要调用函数，经常用于数据的初始化
        :param spider:
        :return:
        '''
        pass
