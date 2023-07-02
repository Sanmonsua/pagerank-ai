import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    pages = corpus.keys()

    linked_pages = corpus[page]
    if not linked_pages:
        for page in pages:
            distribution[page] = 1/(len(corpus))
        return distribution

    for page in pages:

        p = ((1 - damping_factor)/len(corpus))

        if page in linked_pages:
            p += damping_factor/len(linked_pages)

        distribution[page] = p

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = {}
    pages = list(corpus.keys())
    states = []
    
    for _ in range(n):
        if not states:
            next_state = random.choice(pages)
            states.append(next_state)
            continue
        
        state = states[-1]
        distribution = transition_model(corpus, state, damping_factor)
        pages, probabilities = list(zip(*distribution.items()))

        [next_state] = random.choices(pages, weights=probabilities, k=1)
        states.append(next_state)

    for page in pages:
        ranks[page] = states.count(page)/len(states)

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    pages = set(corpus.keys())
    ranks = { page: 1/len(corpus) for page in pages }
    ranking = True

    N = len(pages)
    while ranking:
        new_ranks = {}
        for p in pages:
            num_links = lambda i: (len(corpus[i]) or len(pages))
            linked_from = [i for i in pages if p in corpus[i] or not corpus[i]]

            pr = (1-damping_factor)/N + damping_factor*sum([ranks[i]/num_links(i) for i in linked_from])
            new_ranks[p] = pr

        ranking = any([abs(rank - new_ranks[page]) > 0.001 for page, rank in ranks.items()])
        ranks = new_ranks

    return ranks

if __name__ == "__main__":
    main()
