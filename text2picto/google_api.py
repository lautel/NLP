#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
author: @lcabello

pip install google-api-python-client
pip install search_google

'''
import search_google.api, googleapiclient
import backoff
# About why implementing backoff: https://stackoverflow.com/questions/12471180/frequently-http-500-internal-error-with-google-drive-api-drive-files-get/12640475#12640475


class RequestGoogle:
    def __init__(self):
        # Define buildargs for cse api
        self.buildargs = {
          'serviceName': 'customsearch',
          'version': 'v1',
          'developerKey': '...' 
        }

    @backoff.on_exception(backoff.expo, googleapiclient.errors.HttpError, max_tries=3)
    def send_query(self, word, nresults=1, img_type='color', file_type='jpg'):
        # Define cseargs for search
        '''https://developers.google.com/apis-explorer/?hl=es#p/customsearch/v1/search.cse.list'''
        cseargs = {
          'q': word,
          'cx': '...',
          'cr': 'Spain',
          'searchType': 'image',
          'fileType': file_type,
          'rights': ['cc_publicdomain','cc_noncommercial'],
          'imgColorType': img_type,
          'num': nresults
        }

        # Create a results object
        results = search_google.api.results(self.buildargs, cseargs)

        # Obtain the url links from the search
        # Links are inside results['items'] list
        links = results.get_values('items', 'link')
        return links[0]


if __name__ == '__main__':
    google = RequestGoogle()
    query = "Zaragoza"
    image_url = google.send_query(word=query, nresults=1)
    print(u'Búsqueda en la web: %s \n%s' % (query, image_url))
