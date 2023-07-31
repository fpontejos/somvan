from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CDSView, CustomJS
from bokeh.models import DataTable, TableColumn
from bokeh.transform import factor_cmap, factor_mark
from bokeh.models import Div, Button, Select, TextInput
from bokeh.models import TabPanel, Tabs
from bokeh.models import ColumnDataSource, CDSView, CustomJS, Styles
from bokeh.events import ButtonClick
from sentence_transformers import SentenceTransformer

from bokeh.models import Div

from bokeh.models import AllIndices, GroupFilter, IndexFilter

import os
import pickle
from bokeh.layouts import row, column
import numpy as np
import pandas as pd
from bokeh.io import curdoc

from modules.visualize import *
from modules.constants import *
from modules.plot_components.graph import *

import colorcet as cc 
import json

ROOT_PATH = os.path.dirname(__file__)

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





def get_bmu_byrow(vec_df, som):
    bmu_list = []
    p_ = som.get_weights().shape[2]
    mn = som.get_weights().shape[:2]
    
    vec_vals = vec_df.iloc[:,:p_].values

    for row in vec_vals:
        wx, wy = som.winner(row)
        bmu_list.append(np.ravel_multi_index((wx,wy), mn ) )
    return pd.Series(bmu_list, name='bmu')



def get_meta(meta_path=DASH_CONFIG['data']['meta_csv']):
    
    meta_path_ = [ROOT_PATH] + meta_path.split("/")

    # vectors_df = pd.read_csv(vec_csv_path)
    print(os.path.join(*meta_path_))
    meta_df = pd.read_csv(os.path.join(*meta_path_))
    # combi_df = pd.concat([vectors_df, meta_df], axis=1)
    
    print('load from csv:', meta_path)
        
    return meta_df

def get_vectors(som):
    
    vec_df = pd.read_csv("./data/csv/sbert_vec_df.csv")
    vec_df['bmu'] = get_bmu_byrow(vec_df, som)
    
    p_ = som.get_weights().shape[2]
    vec_vals = vec_df.iloc[:,:p_].values

    hitsmatrix = som.activation_response(vec_vals)

    return vec_df


def get_pickled_som(pickle_path=DASH_CONFIG['data']['som']):
    print('getting pickled som from ', pickle_path)

    # os.path.join(ROOT_PATH, os.path.relpath(os.path.join('.', 'internal', 'models', 'som', 'som.p')))
    pickle_path_ = [ROOT_PATH] + pickle_path.split("/")
    with open(os.path.join(*pickle_path_), 'rb') as infile:
        som = pickle.load(infile)
        print('loading pickled som')
    return som


som = get_pickled_som()

# vec_df = get_vectors(som)

help_body = Div(name="help_body", width=PLOT_WIDTH, width_policy='fixed', css_classes=["help_body"],
text= "<p style='margin-bottom:1em'>" + "</p><p style='margin-bottom:1em'>".join([h for h in  HELP_CONTENTS['um']['body']]) + "</p>"
)

help_title = Div(
text="<h3 style='margin-top:0px'>" + HELP_CONTENTS['um']['title'] + "</h3>"
)

help_col = column(
        help_title,
        help_body,
        name="help_text"
        )



meta_df = get_meta()

json_path_ = [ROOT_PATH] + DASH_CONFIG['data']['topics'].split("/")
json_path = os.path.join(*json_path_)
plot, colorbar_plot, main_source, hex_select, hits_switch, wedge_toggle_switch, to_select, codebook_long, highlight_src, highlight_hex = make_plots(meta_df, som, json_path, HELP_CONTENTS, help_body, help_title)




node_table, table_source, table_view, node_detail = make_tables(
                                            meta_df, 
                                            DASH_CONFIG['details']['table']['meta_cols'], 
                                            DASH_CONFIG['details']['table']['table_cols_attrs']
                                            )

bars, bar_plot, bar_view, bar_index, bar_source = make_bars(meta_df, filter_col='Topic')


G, Gpos, graph_plot, graph_cmap = init_graph(som, meta_df)

graph_plot, edge_source = make_graph_plot(G, Gpos, graph_plot, main_source, graph_cmap)


main_source.selected.js_on_change('indices', 
                                    CustomJS(args=dict(
                                                table_src = table_source,
                                                table_view = table_view,
                                                table_index = IndexFilter(indices=[]),
                                                table_all = AllIndices(),
                                                bar_view = bar_view,
                                                bar_index = bar_index,
                                                bar_source = bar_source,
                                                bar_plot = bar_plot,
                                                edge_src=edge_source,
                                                main_cds=main_source
                                                ),
                                            code=hex_select_cb
                                            ))





#######################
#######################
## topics
# m=15
# n=15
# wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
# hsize = PLOT_HEIGHT/((0.75*n)+0.25)
# size = np.min([hsize,wsize])*.9



#######################
#######################

TOPNN_K = 10
m=15
n=15

query_input = TextInput(placeholder='Search Query', width=200)
codebook_tree   = KDTree(codebook_long)


def get_transformer(modelname='all-MiniLM-L6-v2'):
    """
    Use pre-downloaded model if it exists, 
    otherwise download this model and save it locally.
    """

    models_path_ = [ROOT_PATH] + DASH_CONFIG['data']['vectorizer'].split("/")
    models_path = os.path.join(*models_path_)

    if os.path.exists(models_path):
        print('Using pre-loaded model:', modelname)
        return SentenceTransformer(models_path)
    else:
        print('Downloading transformer model', modelname)
        transformer = SentenceTransformer(modelname)
        transformer.save(models_path)
        return transformer

transformer = get_transformer()



def calculate_query(event):

    topnn = codebook_tree.query(transformer.encode(query_input.value), TOPNN_K)[1].tolist()
    highlight_rank = np.zeros(m*n) #+ .5
    
    rank_alpha = (np.arange(TOPNN_K,0,-1)+1)/(1+TOPNN_K)
    for i in range(TOPNN_K):
        highlight_rank[topnn[i]] = np.max([rank_alpha[i],0.4])

    
    highlight_src.data['rank'] = highlight_rank
    highlight_hex.view = CDSView(filter=IndexFilter(indices=topnn))
    return

def clear_query(event):
    query_input.value = ""
    highlight_hex.view = CDSView(filter=IndexFilter(indices=[]))

button_styles = Styles(flex="1 0 auto")
run_query_button =Button(label="Run Query", button_type="success", styles=button_styles)
clear_query_button =Button(label="Clear Query", styles=button_styles)


run_query_button.on_event(ButtonClick, calculate_query)
clear_query_button.on_event(ButtonClick, clear_query)


footer_input = column(
        Div(text="""<h3 style="margin-top:0px">Enter Query</h3>"""),
        query_input,
        row(run_query_button, 
            clear_query_button, 
                sizing_mode='stretch_width', 
                styles=Styles(justify_content="space-between")
            ),
        # styles=footer_col_style,
        name="footer_input"

        )








#######################
#######################



node_layout = column(
            bar_plot,
            node_detail,
            node_table,
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT
        )

graph_tab = TabPanel(child=graph_plot, title='graph')
node_tab = TabPanel(child=node_layout, title="main", 

                    )

details_tab_layout   = Tabs(tabs=[
                                node_tab,
                                graph_tab, 
                                ], 
                            name="details_tab_layout"
                            )


select_options = column(
        Div(text="""<h3 style="margin-top:0px">Selection Options</h3>"""),
        name="select_options"
        )

curdoc().add_root(select_options) # works
curdoc().add_root(footer_input) # works
curdoc().add_root(help_col) # works
curdoc().add_root(hex_select)

curdoc().add_root(row(hits_switch, name="hits_switch"))
curdoc().add_root(wedge_toggle_switch)

# curdoc().add_root(row(
#     plot, 
#     colorbar_plot,
#     name="plotcolorbar")) # works 

# curdoc().add_root()

curdoc().add_root(to_select) # works
curdoc().add_root(plot) #works
curdoc().add_root(colorbar_plot) 
curdoc().add_root(details_tab_layout)  #works




curdoc().title = "Interactive SOM for Visual Analytics"
