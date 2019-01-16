import scrapy
from ershouche.items import ErshoucheItem
import re


class Ershouche(scrapy.Spider):
    # 定义爬虫的名称，用于在命令中调用
    name = 'ershouche'
    # 定义域名限制，只能爬取xxx域名下的数据
    allowed_domains = ['sh.58.com']

    url_templates = 'https://sh.58.com/ershouche/pn{}/'

    page_num = 70

    # 定义url地址
    start_urls = (url_templates.format(1),)

    def parse(self, response):
        # 调试时取消下面两行代码，并对其他代码进行注释
        # 实际运行时相反
        for num in range(2, self.page_num + 1):
            yield scrapy.Request(self.url_templates.format(num), callback=self.parse)

        info_list = response.xpath("//li[@class='clearfix car_list_less ac_item']")
        for info in info_list:
            item = ErshoucheItem()
            item['brand'] = info.xpath(".//font/text()").extract()[0]
            item['title'] = re.compile('</font> (.*?)\n').findall(info.extract())[0]
            if info.xpath(".//div[@class='first_jx']"):
                item['price'] = info.xpath(".//span[@class='price']/text()").extract()[0]
                item['start_time'] = info.xpath(".//span[@class='year']/text()").extract()[0]
                item['distance'] = info.xpath(".//span[@class='mileage']/text()").extract()[0]
                item['volumn'] = info.xpath(".//span[@class='displacement']/text()").extract()[0]
                item['gear'] = info.xpath(".//span[@class='type']/text()").extract()[0]
            else:
                result = info.xpath(".//div[@class='info_param']/span/text()").extract()
                item['start_time'] = result[0]
                item['distance'] = result[1]
                item['volumn'] = result[2]
                item['gear'] = result[3]
                item['price'] = info.xpath(".//div[@class='col col3']/h3/text()").extract()[0]
            item['tag'] = "_".join(info.xpath(".//div[@class='info_tags']//em/text()").extract())
            status = info.xpath(".//span[@class='tit_icon tit_icon3']/text()").extract()
            item['status'] = status[0] if status else ""
            tmp = info.xpath(".//a[@class='icon_fxc']").extract()
            item['authentication'] = tmp[0] if tmp else ''
            yield item
