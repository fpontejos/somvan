import os


PLOT_WIDTH = 600
PLOT_HEIGHT = 600
SOM_SCALING = "mean"


CONTRAST_COLOR1 = "#ff6361"
CONTRAST_COLOR2 = "#bc5090"

TOPNN_K = 10
BUTTON_STYLES = dict(flex="1 0 auto")


DASH_CONFIG = {

    'data': {
        'meta_csv': 'data/csv/news_meta_tb.csv',
        'topics': 'data/news_topic_labels.json',
        'som':'data/models/som/news_som.p',
        'vectorizer': 'data/models/all-MiniLM-L6-v2',

    },
    'hexagons': {
        'umatrix': {
            'palette': 'Viridis256'
        },
        'hits': {
            'show': True,
            'color': '#FFFFFF'
        },
        'qe': {
            'show': True,
            'palette': 'Magma256'
        },
        'highlight': {
            'show': True,
            'column': 'user',
            'palette': 'Viridis256'
        },
    },
    'details': {
        'table': {
            'show': True,
            'table_cols_attrs' : {
                'meta.title': {'width': 300},
                'meta.topic': {'width': 40},
                'Topic': {'width': 40},
                'topic_name': {'width': 80},
            },
            'meta_cols': ['content', 
                            'Topic',
                            'topic_name',
                            'meta.topic', 
                            'meta.title', 
                            'meta.link', 
                            'bmu']

        }
    }
}

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


HELP_BODY_CONTENTS = "<p style='margin-bottom:1em'>" + \
    "</p><p style='margin-bottom:1em'>".join([h for h in  HELP_CONTENTS['um']['body']]) + \
    "</p>"

HELP_BODY_TITLE = "<h3 style='margin-top:0px'>" + \
    HELP_CONTENTS['um']['title'] + \
    "</h3>"

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
