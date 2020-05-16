
from bs4 import BeautifulSoup
import requests
import requests.exceptions
from urllib.parse import urlsplit
from urllib.parse import urlparse
from bs4.element import Comment
import re
import sys
import os
import subprocess
import argparse
import urllib
import networkx as nx
from pagerank import *
import string
from nltk.stem.porter import PorterStemmer
import os

punctuationList = set(string.punctuation)  # the set of all punctuation signs to skip
stemmer = PorterStemmer()
df = {}
posting = {}

def write(doc_id, text, in_dir):
    '''
    Writes the text upon eliminating all non-ascii characters into a file named doc_id in the in_dir
    '''

    fname = str(doc_id)
    f = open(str(in_dir)+"/"+fname, "w+", encoding="utf-8")
    text = (text.encode('ascii', 'ignore')).decode("utf-8") #ignoring all non ascii characters
    f.write(text)
    f.close()

def tag_visible(element):
    '''
    Returns True if the text in the html tag of the element is visible
    '''
    return not isinstance(element, Comment) \
           and not element.parent.name in ['style', 'script', 'head', 'title', 'meta',
                                                                            '[document]', 'href']

def filter_url(url):

    filter = { '/w/', '#','/File:', '/Wikipedia:', '/Category:', '/Special:', '/Book:', '/Template:'}
    for f in filter:
        if f in url:
            # filtering the url's from wikipedia, only need the content pages
            return True
    return False

def crawler(in_dir, url ="https://en.wikipedia.org/wiki/Computer_science"):
    '''
    Crawls the input url as initial web site and subsequently all the pages that emerge from it by using a queue,
    writes their content in a file in the input directory named after the doc_id, until it has crawled 2000 pages.
    Creates a Graph G having as nodes all the processed pages and directed edges, parent page->child page.


    @param in_dir The directory where to store the content of the crawled web pages
    @param url The initial url to start crawling
    @return G The constructed connected graph used for PageRank
    @return url_map A map from all encountered valid urls to their respective assigned url_nb's
    @return doc_id_map A map from the doc_id(value 1-2000) to the respective url_nb of only the processed written pages
    '''

    G = nx.DiGraph() # a graph to be used for pagerank
    url_map = dict() #keeps a mapping from urls to a url_nb, needed to be able to know if a url was crawled
    doc_id_map = dict() # keeps a map from the doc_id(which is a consecutive int 1-2000) to the url_nb
    doc_id = 1
    url_nb = 1
    try:
        #the initial url
        url_map[url] = url_nb
        G.add_node(url_nb)
        # a queue of urls to be crawled
        new_urls = []
        new_urls.append(url)
        # to keep track of already crawled urls, and avoid duplicates
        processed_urls = set()
        # a set of domains inside the target website(wikipedia)
        found_local_urls = set()
        # keeps the broken urls
        bad_urls = set()
        url_nb += 1

        while len(new_urls):
            # only need 2000 docs
            if doc_id > 2000:
                break
            url = new_urls.pop(0)
            url_parent = url_map[url]
            doc_id_map[doc_id] = url_parent
            processed_urls.add(url)
            # getting the url's content
            try:
                response = requests.get(url)
            except :
                bad_urls.add(url)
                continue

            # extract base url to deal with only local links
            parts = urlsplit(url)
            base = "{0.netloc}".format(parts)
            strip_base = base.replace("www.", "")
            base_url = "{0.scheme}://{0.netloc}".format(parts)
            path = url[:url.rfind('/') + 1] if '/' in parts.path else url
            if response.status_code != 200:
                # if the response raises an error, skip that url
                continue
            if filter_url(response.url):
                continue
            print("Processing %s" % url)
            # create a beautiful soup for the html document
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.findAll(text=True)
            visible_texts = filter(tag_visible, texts) # get the visible text from the page
            write( doc_id, u" ".join(t.strip() for t in visible_texts), in_dir) # writing the text from the page onto a file
            doc_id += 1
            # iterating through all the links on the page
            for link in soup.find_all('a'):
                # extract link url from the anchor
                anchor = link.attrs["href"] if "href" in link.attrs else ''
                if anchor.startswith('/'):
                    local_link = base_url + anchor

                elif strip_base in anchor:
                    local_link = anchor

                elif not anchor.startswith('http'):
                    local_link = path + anchor

                else:
                    continue
                #need to check if this page was already given a url_nb
                if filter_url(local_link):
                    continue

                found_local_urls.add(local_link)
                if (local_link in url_map):
                    url_child = url_map[local_link]
                else:
                    url_map[local_link] = url_nb
                    url_child = url_nb
                    url_nb += 1

                G.add_edge(url_parent, url_child) #creating a directed edge in the graph

            for i in found_local_urls:
                if not i in new_urls and not i in processed_urls: #to avoid duplicates and self loops
                    new_urls.append(i)
        return G, url_map, doc_id_map
    except KeyboardInterrupt:
        sys.exit()


def main():
    print("main")
    crawler()

if __name__ == "__main__":
    main()