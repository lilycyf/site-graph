import time
from bs4 import BeautifulSoup
import urllib
import requests
from pyvis.network import Network
import networkx as nx
import argparse
import pickle
import scipy
import numpy as np


import os
import random
import string

from pydantic import BaseModel
from openai import OpenAI

from collections import deque

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

INTERNAL_COLOR = '#0072BB'
EXTERNAL_COLOR = '#FF9F40'
ERROR_COLOR = '#FF0800'
RESOURCE_COLOR = '#2ECC71'

NOT_VISITED_COLOR = 'gray'



def handle_error(error, error_obj, r, url, visited, error_codes):
    error = str(error_obj) if error else r.status_code
    visited.add(url)
    error_codes[url] = error
    print(f'{error} ERROR while visiting {url}')


def has_been_visited(url, visited):
    return url in visited or url.rstrip('/') in visited or url + '/' in visited

def handle_page_text(page):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')

    # Remove unwanted elements
    # Remove the header section
    for header in soup.find_all(['header', 'nav', 'div'], {'class': ['header', 'nav', 'site-header', 'top-header']}):
        header.decompose()

    # Remove the footer section
    for footer in soup.find_all(['footer', 'div'], {'class': ['footer', 'site-footer', 'bottom-footer']}):
        footer.decompose()

    # Remove sidebars (common layout elements that are not part of the main content)
    for sidebar in soup.find_all(['aside', 'div'], {'class': ['sidebar', 'sidebar-content', 'widget']}):
        sidebar.decompose()

    # Remove advertisements (common ad classes)
    for ad in soup.find_all(['div', 'section', 'iframe'], {'class': ['ad', 'adsbygoogle', 'advertisement']}):
        ad.decompose()

    # Remove any script or style elements
    for script in soup(['script', 'style']):
        script.decompose()

    # Get text from the remaining content, preserving newlines
    page_text = soup.get_text(separator='\n')

    # Clean up excessive whitespace while keeping each line separate
    cleaned_text = '\n'.join(line.strip() for line in page_text.splitlines() if line.strip())

    # Return the final cleaned text
    return cleaned_text



def crawl(url, visit_external, keep_queries, args, filter, question):
    visited = set()
    edges_with_labels = {}  # Dictionary to store edges with labels
    resource_pages = set()
    error_codes = dict()
    redirect_target_url = dict()
    all_text = ""

    # Folder setup
    target_folder = 'data/'

    # Generate random ID and check if folder exists
    while True:
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        folder_path = os.path.join(target_folder, random_id)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            break
    
    screenshot_folder = os.path.join(folder_path, 'screenshot')
    html_folder = os.path.join(folder_path, 'html')

    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)
    if not os.path.exists(html_folder):
        os.makedirs(html_folder)


    head = requests.head(url, timeout=10)
    site_url = head.url
    redirect_target_url[url] = site_url

    to_visit = deque()
    to_visit.append((site_url, None))

    counter = 0

    while to_visit:
        url, from_url = to_visit.pop()

        print('Visiting', url, 'from', from_url)

        error = False
        error_obj = None
        try:
            page = requests.get(url, timeout=10)
            # Collect text content
            page_text = handle_page_text(page)
            text = f"==============================\nURL: {url}\n------------------------------\nContent: {page_text}\n==============================\n"

            all_text += text

            full_path = os.path.join(screenshot_folder, 'screenshot.png')

            # Check if file exists, if it does, append a number to the filename
            if os.path.exists(full_path):
                counter = 1
                while os.path.exists(os.path.join(screenshot_folder, f'screenshot{counter}.png')):
                    counter += 1
                full_path = os.path.join(screenshot_folder, f'screenshot{counter}.png')

            take_full_screenshot(url, full_path)
            print(f"Full-page screenshot saved as {full_path}")


            full_path = os.path.join(html_folder, "html.html")

            # Check if file exists, if it does, append a number to the filename
            if os.path.exists(full_path):
                counter = 1
                while os.path.exists(os.path.join(html_folder, f'html{counter}.html')):
                    counter += 1
                full_path = os.path.join(html_folder, f'html{counter}.html')

            html = page.text
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(html)
            print(f"HTML content saved to {full_path}")


        except requests.exceptions.RequestException as e:
            error = True
            error_obj = e

        if error or not page:
            handle_error(error, error_obj, page, url, visited, error_codes)
            continue
        
        # Don't look for links in external pages
        if not (visit_external or url.startswith(site_url)):
            continue

        soup = BeautifulSoup(page.text, 'html.parser')

        # Handle <base> tags
        base_url = soup.find('base')
        base_url = '' if base_url is None else base_url.get('href', '')

        to_visit_edges_temp = {}
        link_list = soup.find_all('a', href=True)

        for link in link_list:
            link_url = link['href']
            link_text = link.get_text(strip=True)  # Get the text of the link

            # Get the raw HTML content of the page
            raw_html = str(soup)

            # Find the position of the link in the raw HTML
            link_html = str(link)  # Convert the link object to a string (HTML of the <a> tag)
            link_pos = raw_html.find(link_html)  # Find the position of the link in the raw HTML

            # Get 100 characters before and after the link, ensuring bounds are within the string
            start_pos = max(0, link_pos - 100)  # Ensure start_pos is not negative
            end_pos = min(len(raw_html), link_pos + len(link_html) + 100)  # Ensure end_pos is within bounds

            # Extract the surrounding text
            surrounding_text = raw_html[start_pos:end_pos]
            
            to_visit_edges_temp[(url, link_url)] = (link_text, surrounding_text) or ("No Label", surrounding_text)

        
        if filter:
            link_list = filter_edges(to_visit_edges_temp, args, link_list, question)


        for link in link_list:
            link_url = link['href']

            if link_url.startswith('mailto:'):
                continue
            
            # Resolve relative paths
            if not link_url.startswith('http'):
                link_url = urllib.parse.urljoin(url, urllib.parse.urljoin(base_url, link_url))

            # Remove queries/fragments from internal links
            if not keep_queries and (visit_external or link_url.startswith(site_url)):
                link_url = urllib.parse.urljoin(link_url, urllib.parse.urlparse(link_url).path)

            # Load where we know that link_url will be redirected
            if link_url in redirect_target_url:
                link_url = redirect_target_url[link_url]

            if not has_been_visited(link_url, visited) and (visit_external or link_url.startswith(site_url)):
                is_html = False
                error = False
                error_obj = None

                try:
                    head = requests.head(link_url, timeout=10)
                    if head and 'html' in head.headers.get('content-type', ''):
                        is_html = True
                except requests.exceptions.RequestException as e:
                    error = True
                    error_obj = e

                if error or not head:
                    handle_error(error, error_obj, head, link_url, visited, error_codes)
                    edges_with_labels[(url, link_url)] = (link_text, surrounding_text) or ("No Label", surrounding_text)
                    continue

                visited.add(link_url)
                
                redirect_target_url[link_url] = head.url
                link_url = redirect_target_url[link_url]
                visited.add(link_url)

                if is_html:
                    if (visit_external or link_url.startswith(site_url)):
                        to_visit.append((link_url, url))
                else:
                    resource_pages.add(link_url)
            
            # Add edge with label
            edges_with_labels[(url, link_url)] = (link_text, surrounding_text) or ("No Label", surrounding_text)
        print(to_visit)
        counter += 1

        # Check if have enough information to answer the question
        if len(to_visit) != 0 and counter % 3 == 0:
            
            client = OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Please answer the question based on the reference information"},
                    {"role": "user", "content": f"Question: {question}\n\nReference: {all_text}"}
                ]
            )

            output = completion.choices[0].message.content
            print(output)
            while True:
                satisfy = input(f"Are you satisfied with the current answer to your question: '{question}'? If no, we will look for more information. (Y/N): ").strip().upper()
                if satisfy != "Y" and satisfy != "N":
                    print("Invalid input. Please enter 'Y' or 'N'.")
                else:
                    break
            if satisfy == "Y":
                break
            elif satisfy == "N":
                pass
    
        
    with open(os.path.join(folder_path, "output.txt"), "w") as file:
        # Write the string to the file
        file.write(all_text)

    print(f"Content has been saved to {os.path.join(folder_path, 'output.txt')}")

    return edges_with_labels, error_codes, resource_pages


def get_node_info(nodes, error_codes, resource_pages, args, visited_pages):
    node_info = []
    for node in nodes:
        if node in visited_pages:
            if node in error_codes:
                node_info.append(f'Error: {error_codes[node]}; visited')
            elif node in resource_pages:
                node_info.append('resource; visited')
            elif node.startswith(args.site_url):
                node_info.append('internal; visited')
            else:
                node_info.append('external; visited')
        else:
            if node in error_codes:
                node_info.append(f'Error: {error_codes[node]}')
            elif node in resource_pages:
                node_info.append('resource')
            elif node.startswith(args.site_url):
                node_info.append('internal')
            else:
                node_info.append('external')
    return node_info

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def take_full_screenshot(url, file_name, max_scrolls=20):
    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    
    # Create a WebDriver instance
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    time.sleep(5)  # Allow the page to load completely

    # Determine the total height by scrolling
    scroll_height = driver.execute_script("return window.innerHeight")
    total_height = driver.execute_script("return document.body.scrollHeight")
    previous_height = 0
    scroll_count = 0
    new_content_loaded = True

    while new_content_loaded and scroll_count < max_scrolls:
        # Scroll to the bottom of the current visible portion
        driver.execute_script(f"window.scrollTo(0, {scroll_count * scroll_height});")
        time.sleep(3)  # Wait for potential content loading

        # Check if new content was loaded by comparing heights
        current_height = driver.execute_script("return document.body.scrollHeight")
        new_content_loaded = current_height > total_height
        total_height = current_height if new_content_loaded else total_height
        scroll_count += 1

    # Set the window size to the full page height
    driver.set_window_size(driver.execute_script("return document.body.scrollWidth"), total_height)

    # Take a single full-page screenshot
    driver.save_screenshot(file_name)
    
    # Close the browser
    driver.quit()



class need_visited(BaseModel):
    output: list[str]


def filter_edges(edges_with_labels, args, link_list, question):

    # Start by iterating over each edge
    info_lst = [info for _, info in edges_with_labels.items()]
    # edge_lst = [edge for edge, _ in edges_with_labels.items()]


    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a smart crawler assistant, and users will give you a question and a list of pairs. The first item in each pair of the list represents the link_text of a link, and the second item represents the rounding_text of the link. You need to select between: 'Yes', 'No', 'Not Sure' on whether this link might able to contain the information that can answer the question. You need to return a list of equal length to the list provided by the user."},
            {"role": "user", "content": f"Question: {question}\n\nLink Pairs: {info_lst}"}
        ],
        response_format=need_visited,
    )

    output = completion.choices[0].message.parsed
    print(output)

    # want_visit_edge = [edge for edge, score in zip(edge_lst, output.output) if score == 'Yes']
    want_visit_edge = [link for link, score in zip(link_list, output.output) if score == 'Yes']

    return want_visit_edge



def collect_info_from_url(url):
    print('Visiting', url)
    page = requests.get(url, timeout=10)
    # Collect text content
    page_text = handle_page_text(page)
    text = f"==============================\nURL: {url}\n------------------------------\nContent: {page_text}\n==============================\n"
    html = page.text

    return text, html


def collect_info_from_urls(url_lst):
    all_text = ""
    # Now, go over each visited page 
    for i, url in enumerate(url_lst):
        text, html = collect_info_from_url(url)

        all_text += text

        screenshot_filename = f"screenshot/screenshot_{i}.png"
        take_full_screenshot(url, screenshot_filename)
        print(f"Full-page screenshot saved as {screenshot_filename}")

        html_filename = f"html/html_{i}.html"
        with open(html_filename, 'w', encoding='utf-8') as file:
            file.write(html)
        print(f"HTML content saved to {html_filename}")

        
    with open("output.txt", "w") as file:
        # Write the string to the file
        file.write(all_text)

    print("Content has been saved to output.txt")



def visualize(edges_with_labels, error_codes, resource_pages, args, visited_pages=-1):
    G = nx.DiGraph()

    # Add edges with labels to the graph
    for (u, v), (link_text, surrounding_text) in edges_with_labels.items():
        label = link_text
        G.add_edge(u, v, label=label)

    # Contract any extra nodes 
    nodes = set(G.nodes)
    for node in nodes:
        alias = node + '/'
        if alias in nodes:
            print(f'Contracting {node} and {alias}')
            G = nx.contracted_nodes(G, alias, node)

    if args.save_txt is not None or args.save_npz is not None:
        nodes = list(G.nodes())
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes, dtype=int)

        if args.save_npz is not None:
            base_fname = args.save_npz.replace('.npz', '')
            scipy.sparse.save_npz(args.save_npz, scipy.sparse.coo_matrix(adj_matrix))
        else:
            base_fname = args.save_txt.replace('.txt', '')
            np.savetxt(args.save_txt, adj_matrix, fmt='%d')

        node_info = get_node_info(nodes, error_codes, resource_pages, args, []) if visited_pages == -1 else get_node_info(nodes, error_codes, resource_pages, args, visited_pages)
        with open(base_fname + '_nodes.txt', 'w') as f:
            f.write('\n'.join([nodes[i] + '\t' + node_info[i] for i in range(len(nodes))]))

    # Create a pyvis network graph
    net = Network(width=args.width, height=args.height, directed=True)
    net.from_nx(G)

    # Apply button settings or load options if provided
    if args.show_buttons:
        net.show_buttons()
    elif args.options is not None:
        try:
            with open(args.options, 'r') as f:
                net.set_options(f.read())
        except FileNotFoundError as e:
            print('Error: options file', args.options, 'not found.')
        except Exception as e:
            print('Error applying options:', e)

    # Customize nodes based on error codes, site URLs, and other criteria
    for node in net.nodes:
        node['size'] = 15
        node['label'] = ''
        if node['id'].startswith(args.site_url):
            node['color'] = INTERNAL_COLOR
            if node['id'] in resource_pages:
                node['color'] = RESOURCE_COLOR
        else:
            node['color'] = EXTERNAL_COLOR
        
        if visited_pages != -1 and node['id'] not in visited_pages:
            node['color'] = NOT_VISITED_COLOR

        if node['id'] in error_codes:
            node['title'] = f'{error_codes[node["id"]]} Error: <a href="{node["id"]}">{node["id"]}</a>'

            if not args.only_404 or error_codes[node['id']] == 404:
                node['color'] = ERROR_COLOR
        else:
            node['title'] = f'<a href="{node["id"]}">{node["id"]}</a>'
    
    # Add edge labels
    for edge in net.edges:
        u, v = edge['from'], edge['to']
        edge_label = G.edges[u, v].get('label', '')
        edge['title'] = edge_label  # Add edge label as the title

    # Remove saved contractions (otherwise save_graph crashes)
    for edge in net.edges:
        edge.pop('contraction', None)

    # Save the graph to file
    if visited_pages == -1:
        net.save_graph(args.vis_file)
    else:
        net.save_graph(args.vis_file_visited)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the link graph of a website.')
    parser.add_argument('site_url', type=str, help='the base URL of the website', nargs='?', default='')

    # Defaults
    vis_file = 'site.html'
    vis_file_visited = 'site_visited.html'
    data_file = 'crawl.pickle'
    width = 1000
    height = 800

    parser.add_argument('--vis-file', type=str, help=f'filename in which to save HTML graph visualization (default: {vis_file})', default=vis_file)
    parser.add_argument('--vis-file-visited', type=str, help=f'filename in which to save HTML graph visualization with site visited (default: {vis_file_visited})', default=vis_file_visited)
    parser.add_argument('--data-file', type=str, help=f'filename in which to save crawled graph data (default: {data_file})', default=data_file)
    parser.add_argument('--width', type=int, help=f'width of graph visualization in pixels (default: {width})', default=width)
    parser.add_argument('--height', type=int, help=f'height of graph visualization in pixels (default: {height})', default=height)
    parser.add_argument('--visit-external', action='store_true', help='detect broken external links (slower)')
    parser.add_argument('--show-buttons', action='store_true', help='show visualization settings UI')
    parser.add_argument('--options', type=str, help='file with drawing options (use --show-buttons to configure, then generate options)')
    parser.add_argument('--from-data-file', type=str, help='create visualization from given data file', default=None)
    parser.add_argument('--force', action='store_true', help='override warnings about base URL')
    parser.add_argument('--save-txt', type=str, nargs='?', help='filename in which to save adjacency matrix (if no argument, uses adj_matrix.txt). Also saves node labels to [filename]_nodes.txt', const='adj_matrix.txt', default=None)
    parser.add_argument('--save-npz', type=str, nargs='?', help='filename in which to save sparse adjacency matrix (if no argument, uses adj_matrix.npz). Also saves node labels to [filename]_nodes.txt',  const='adj_matrix.npz', default=None)
    parser.add_argument('--keep-queries',  action='store_true', help='create visualization from given data file')
    parser.add_argument('--only-404', action='store_true', help='only color 404 error nodes in the error color')

    args = parser.parse_args()

    if args.from_data_file is None:
        if not args.site_url.startswith('https'):
            if not args.force:
                print('Warning: not using https. If you really want to use http, run with --force')
                exit(1)
        
        # Request a new argument from the terminal
        question = input("Please tell me what question you hope to answer from the information gain from this crawler: ")

        edges, error_codes, resource_pages = crawl(args.site_url, args.visit_external, args.keep_queries, args, True, question)
        print('Crawl complete.')

        with open(args.data_file, 'wb') as f:
            pickle.dump((edges, error_codes, resource_pages, args.site_url), f)
            print(f'Saved crawl data to {args.data_file}')
    else:
        with open(args.from_data_file, 'rb') as f:
            edges, error_codes, resource_pages, site_url = pickle.load(f)
            args.site_url = site_url

    visualize(edges, error_codes, resource_pages, args)
    # url_lst = filter_edges(edges, args)
    # visualize(edges, error_codes, resource_pages, args, url_lst)
    print('Saved graph to', args.vis_file)
