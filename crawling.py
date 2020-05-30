import requests
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os

def crawling():
    except_count = 0
    except_list = []
    if not os.path.exists('lyrics'):
        os.mkdir('lyrics')

    genre_list = ['christian', "country-music", "hip-hop-rap", "pop", "rhythm-blues", "rock"]
    driver = webdriver.Chrome('C:/chromedriver.exe')
    for genre in genre_list:
        if not os.path.exists('lyrics\\' + genre):
            os.mkdir('lyrics\\' + genre)

        driver.get("http://www.songlyrics.com/news/top-genres/" + genre + "/")
        element = driver.find_element_by_class_name("tracklist")
        td_tag = element.find_elements_by_tag_name("td")

        title_list = []
        for i in range(6, len(td_tag), 3):
            try:
                title = str(td_tag[i - 1].find_element_by_tag_name('a').text)[:-7]

                link = td_tag[i - 1].find_element_by_tag_name('a').get_attribute('href')
                html = requests.get(link).text
                bs = BeautifulSoup(html, 'html.parser')

                lyrics = str(bs.find('p', id="songLyricsDiv"))
                re_br = re.compile('<br/>')
                re_all = re.compile('<.*>')
                lyrics = re.sub(re_br, '', lyrics)
                lyrics = re.sub(re_all, '', lyrics)
                time.sleep(0.5)

                if lyrics == "None" or lyrics == " " or lyrics == "":
                    continue  # 간혹 링크가 사라진 노래가 존재

                title_list.append(title + "\n")
                print(title)
                print(lyrics)
                print(type(lyrics))
                text = open('lyrics\\' + genre + "\\" + title + ".txt", 'w', -1, "utf-8")
                text.write(lyrics)
                text.close()
                print("--------------------------------------------")

            except Exception as ex:
                except_count += 1
                except_list.append(str(ex))
                pass

    print(except_count)
    print(except_list)


crawling()