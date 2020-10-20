from scrapy.http import HtmlResponse
from selenium import webdriver


class Selenium(object):

    @classmethod
    def process_request(cls, request, spider):
        option_chrome = webdriver.ChromeOptions()
        prefs = {
            'profile.default_content_setting_values': {
                'images': 2,  # 禁止加载图片
                'javascript': 2   # 禁止执行js
            }
        }
        option_chrome.add_experimental_option('prefs', prefs)
        # option_chrome.add_argument('--headless')
        # option_chrome.add_argument('--no-startup-window')
        option_chrome.add_argument('--no-sandbox')
        driver = webdriver.Chrome(chrome_options=option_chrome)
        driver.get(request.url)
        content = driver.page_source.encode('utf-8')
        driver.quit()
        return HtmlResponse(request.url, encoding='utf-8', body=content, request=request)
