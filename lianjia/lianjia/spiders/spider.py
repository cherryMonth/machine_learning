import scrapy
from lianjia.items import LianjiaItem
import re

url_template = 'https://sh.lianjia.com/zufang/pg{}/#contentList'


class WeBook(scrapy.Spider):
    # 定义爬虫的名称，用于在命令中调用
    name = 'lianjia'
    # 定义域名限制，只能爬取xxx域名下的数据
    allowed_domains = ['sh.lianjia.com']

    # 定义url地址
    start_urls = (
        'https://sh.lianjia.com/zufang/',
    )

    def parse(self, response):
        job_list = response.css('.content__list--item')
        item = LianjiaItem()
        for page in job_list:
            item['title'] = page.css('.content__list--item--title').xpath('.//a/text()').extract()[0].replace(' ',
                                                                                                              '').replace(
                '\n', '')
            region = page.css('.content__list--item--des').xpath('.//a/text()').extract()

            item['region'] = region[0]
            item['sub_region'] = region[1]

            label_tmp = page.css('.content__list--item--des').xpath(".//text()").extract()
            label_tmp = list(
                map(lambda x: x.replace(' ', '').replace('\n', '').replace('/', '').replace('-', ''), label_tmp))
            label_tmp = [label for label in label_tmp if label]
            item['region'] = label_tmp[0]
            item['sub_region'] = label_tmp[1]
            item['area'] = float(re.sub("\D", "", label_tmp[2]))
            item['location'] = label_tmp[3]
            item['room'] = int(list(re.sub("\D", "", label_tmp[4]))[0])
            item['hall'] = int(list(re.sub("\D", "", label_tmp[4]))[1])
            item['toilet'] = int(list(re.sub("\D", "", label_tmp[4]))[2])
            item['layer'] = label_tmp[5].split('（')[0]
            item['floor'] = int(re.sub("\D", "", label_tmp[5].split('（')[1]))
            item['price'] = page.css('.content__list--item-price').xpath('.//text()').extract()[0]
            item['tenancy'] = page.css('.content__list--item-price').xpath('.//text()').extract()[1].strip()
            item['brand'] = page.css('.content__list--item--brand').xpath('.//text()').extract()[0].replace(' ',
                                                                                                            '').replace(
                '\n', '')
            item['time'] = page.css('.content__list--item--time').xpath('.//text()').extract()[0].strip()
            tag_tmp = page.css('.content__list--item--bottom').xpath('.//i/text()').extract()
            item['tag'] = '_'.join(tag_tmp)
            print(item)
            yield item

        for key in range(2, 100):
            print("正在爬取{}".format(key))
            # 取前10页
            print(url_template.format(key))
            yield scrapy.Request(url_template.format(key), callback=self.parse)
