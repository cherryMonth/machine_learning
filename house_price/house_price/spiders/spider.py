import scrapy
from house_price.items import HousePriceItem

url_templates = 'http://data.eastmoney.com/cjsj/hpdetail.aspx?date={}-{}-1'
url_list = [url_templates.format(year, month) for year in range(2011, 2019) for month in range(1, 13)]


class Houseprice(scrapy.Spider):
    # 定义爬虫的名称，用于在命令中调用
    name = 'house_price'
    # 定义域名限制，只能爬取xxx域名下的数据
    allowed_domains = ['data.eastmoney.com']

    # 定义url地址
    start_urls = ('http://data.eastmoney.com/cjsj/hpdetail.aspx?date=2011-1-1',)

    page_num = 0

    def parse(self, response):
        # 调试时取消下面两行代码，并对其他代码进行注释
        # 实际运行时相反

        info_list = response.xpath(".//tr[@class='']")
        for info in info_list:
            item = HousePriceItem()
            result = info.xpath('.//td/text()').extract()
            item['time'] = result[0].strip()
            item['city'] = result[1].strip()
            item['residence_MOM'] = result[2].strip()
            item['residence_YOY'] = result[3].strip()
            item['residence_FBI'] = result[4].strip()
            item['commercial_residence_MOM'] = result[5].strip()
            item['commercial_residence_YOY'] = result[6].strip()
            item['commercial_residence_FBI'] = result[7].strip()
            item['second_hand_MOM'] = result[8].strip()
            item['second_hand_YOY'] = result[9].strip()
            item['second_hand_FBI'] = result[10].strip()
            yield item
        self.page_num += 1
        if self.page_num < len(url_list):
            yield scrapy.Request(url_list[self.page_num], callback=self.parse)
        else:
            yield None