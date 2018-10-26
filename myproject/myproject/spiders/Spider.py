import scrapy
from myproject.items import MyprojectItem

position = ["北京", "天津", "上海", "重庆", "河北", "山西", "辽宁",
            "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西",
            "山东", "河南", "湖北", "湖南","广东", "海南", "四川",
            "贵州", "云南", "陕西", "甘肃", "青海", "台湾", "内蒙古",
            "广西", "西藏", "宁夏", "新疆", "香港", "澳门"]

url_template = 'http://piao.qunar.com/ticket/list.htm?keyword={}' \
               '&region=&from=mpl_search_suggest&page={}'


class WeBook(scrapy.Spider):
    # 定义爬虫的名称，用于在命令中调用
    name = 'hot'
    # 定义域名限制，只能爬取xxx域名下的数据
    allowed_domains = ['piao.qunar.com']

    # 定义url地址
    start_urls = (
        'http://piao.qunar.com/ticket/list.htm?keyword=北京&region=&from=mpl_search_suggest&page=1',
    )

    def parse(self, response):
        job_list = response.css('.sight_item')
        item = MyprojectItem()
        for page in job_list:
            item['name'] = page.xpath(".//a[@class='name']/text()").extract()[0]
            level_handler = page.xpath(".//span[@class='level']/text()").extract()
            item['level'] = level_handler[0] if level_handler else ''
            item['hot'] = page.xpath(".//span[@class='product_star_level']/em/span/text()").extract()[0][3:]
            item['area'] = "[" + page.css('.area').xpath(".//a/text()").extract()[0] + "]"
            item['address'] = page.css(".address").xpath(".//span/text()").extract()[0]
            price_temp = page.css(".sight_item_price").xpath(".//em/text()").extract()
            item['price'] = price_temp[0] if price_temp else 0
            item['info'] = page.css(".intro").xpath("./text()").extract()[0]
            num_handler = page.xpath(".//span[@class='hot_num']/text()").extract()
            item['num'] = num_handler[0] if num_handler else 0
            yield item

        for key in position:
            print("正在爬取{}".format(key))
            # 取前10页
            for page in range(1, 14):
                print("正在爬取第{}页".format(page))
                yield scrapy.Request(url_template.format(key, page), callback=self.parse)
