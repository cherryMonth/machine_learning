import scrapy
from douban.items import DoubanItem
import re
import json


class Douban(scrapy.Spider):
    # 定义爬虫的名称，用于在命令中调用
    name = 'douban'
    # 定义域名限制，只能爬取xxx域名下的数据
    allowed_domains = ['movie.douban.com']

    cookies = {
        'bid': 'YFvyGCpE3Ts',
        "ll": "108296",
        "douban-fav-remind": "1",
        "viewed": "30167776",
        "ap_v": "0,6.0"
    }

    url_templates = 'https://movie.douban.com/j/new_search_subjects?sort=T&range=0,10&tags=%E7%83%82%E7%89%87&start={}'

    page_num = 0

    # 定义url地址
    start_urls = (url_templates.format(page_num),)

    def parse(self, response):
        # 调试时取消注释下面两行代码，并对其他代码进行注释
        # 实际运行时相反
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        if 'tag' in response.url:
            url_data = json.loads(response.body[84:-20].decode())['data']
            print(url_data)
            self.page_num += 20
            for data in url_data:
                yield scrapy.Request(data['url'], cookies=self.cookies, callback=self.parse)
            if self.page_num == 1000:
                yield None
            else:
                yield scrapy.Request(self.url_templates.format(self.page_num), cookies=self.cookies,
                                     callback=self.parse)
        else:
            info = response.xpath('//div[@id="info"]')[0]
            if '集数' in info.extract() and '单集片长' in info.extract():  # 对于电视剧类型选择丢弃
                return None
            else:
                item = DoubanItem()
                item['name'] = response.xpath('//span[@property="v:itemreviewed"]/text()').extract()[0]
                item['douban_id'] = response.url.split("/")[-2]
                star = response.xpath('//span[@class="attrs"]/span/a/text()').extract() or response.xpath(
                    '//span[@class="attrs"]/a/text()').extract()
                item['star'] = '_'.join(star)
                item['director'] = info.xpath('//span[@class="attrs"]/a')[0].xpath('./text()').extract()[0]
                item['screenwriter'] = info.xpath('//span[@class="attrs"]/a')[1].xpath('./text()').extract()[0]
                item['tag'] = info.xpath('//span[@property="v:genre"]/text()').extract()[0]
                item['region'] = re.compile(r'<span class="pl">制片(.*?):</span> (.*?)<br>').findall(info.extract())[0][1]
                item['language'] = re.compile(r'<span class="pl">语言(.*?):</span> (.*?)<br>').findall(info.extract())[0][
                    1]
                item['release_time'] = "_".join(info.xpath('//span[@property="v:initialReleaseDate"]/text()').extract())
                item['film_length'] = info.xpath('//span[@property="v:runtime"]/text()').extract()[0]
                item['collections'] = info.xpath('//span[@property="v:votes"]/text()').extract()[0]
                stars_info = info.xpath("//div[@class='ratings-on-weight']//span[@class='rating_per']/text()").extract()
                item['five_star'] = round(float(stars_info[0][:-1]) / 100, 4)
                item['four_star'] = round(float(stars_info[1][:-1]) / 100, 4)
                item['three_star'] = round(float(stars_info[2][:-1]) / 100, 4)
                item['two_star'] = round(float(stars_info[3][:-1]) / 100, 4)
                item['one_star'] = round(float(stars_info[4][:-1]) / 100, 4)
                item['intro'] = response.xpath('//span[@property="v:summary"]/text()').extract()[0].strip()
                item['rating_num'] = info.xpath("//strong[@class='ll rating_num']/text()").extract()[0]
                yield item
