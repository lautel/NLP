#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import os, json
import backoff

def fatal_code(e):
    return 400 <= e.response.status_code < 500


class RequestAPI:
    def __init__(self, language, catalog):
        self.language = language
        self.catalog = catalog

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3, giveup=fatal_code)
    def send_query(self, word, nresults, TXTlocate):
        variable = 'language=%s&word=%s&catalog=%s&nresults=%d&TXTlocate=%d' % \
                        (self.language, word, self.catalog, nresults, TXTlocate)
        url = 'http://........' % variable
        my_response = requests.Session().get(url, verify=True)

        # For successful API call, response code will be 200 (OK)
        if my_response.ok:
            # Loading the response data into a dict variable
            # json.loads takes in only binary or string variables so using content to fetch binary content
            # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
            jData = json.loads(my_response.content)
            jData_final = []
            # Filtro palabras que me devuelve la busqueda pero no son relevantes. Por ejemplo: pez - pezuña, cuidado - cuidadora
            for symbol in jData['symbols']:
                if symbol['name'].lower().split()[0] == word:
                    jData_final.append(symbol)
                elif '_'.join(symbol['name'].lower().split()) == word:
                    jData_final.append(symbol)
            return jData_final
        else:
            # If response code is not ok (200), print the resulting http error code with description
            my_response.raise_for_status()
            return


class ImageScraper:
    def __init__(self, download_path):
        self.download_path = download_path
        self.session = requests.Session()

    def scrape_images(self, image_url):
        image_name = image_url.split('/')[-1]
        self.save_image(image_name, image_url)
        return image_name

    def save_image(self, file_name, item_link):
        response = self.session.get(item_link, stream=True)

        if response.status_code == 200:
            with open(os.path.join(self.download_path, file_name), 'wb') as image_file:
                for chunk in response.iter_content(1024):
                    image_file.write(chunk)
        # print(file_name)
