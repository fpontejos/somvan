import os


PLOT_WIDTH = 600
PLOT_HEIGHT = 600
SOM_SCALING = "mean"


CONTRAST_COLOR1 = "#ff6361"
CONTRAST_COLOR2 = "#bc5090"


HELP_CONTENTS = {
    'um': {
        'title': "UMatrix View",
        'body': ["The color of the hexagons denote the average distance of each node to its neighbors.",
        "Darker nodes are closer to their neighbors, while yellow nodes are farther away."
        ]
        },
    'tp': {
        'title': "Topic Model View",
        'body': ["Each node is assigned the color of the most frequently occuring topic abong the documents in the node.",
        "The wedges in the middle denotes the percentage of documents in the node that belong to this topic. Full circle wedges mean all the documents in that node belong to the same topic."
        ]
    },
    'to': {
        'title': 'Topic Overlay View',
        'body': ["Choosing a specific topic from the selection highlights the nodes that contain documents with that topic.",
        "Darker nodes correspond to a higher percentage of documents in the node containing the topic."]
    },
    'rc': {
        'title': "Temporal Overlay",
        'body': ["The color of the nodes correspond to the average recency of its documents.",
        "Red nodes are on average more recent than blue nodes."]
    }
}

def _embed_js_contents(p, filename):
    js_path = os.path.join(p, 'modules', 'js', filename)
    print(js_path)
    with open(js_path) as f:
        return f.read()

def get_callbacks(p):

    cb = {}

    cb['topic_dropdown'] = _embed_js_contents(p, 'topic_dropdown.js')

    cb['hex_dropdown'] = _embed_js_contents(p, 'hex_dropdown.js')

    cb['hex_click'] = _embed_js_contents(p, 'hex_click.js')

    cb['row_click'] = _embed_js_contents(p, 'row_click.js')

    print(cb.keys())

    return cb 
