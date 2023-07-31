import numpy as np
import pandas as pd
import json
import colorcet as cc

from scipy.spatial import KDTree
from collections import Counter

from sklearn.preprocessing import MinMaxScaler

from bokeh.layouts import row, column
from bokeh.models import LinearColorMapper
from bokeh.models import ColorBar, HoverTool, BasicTicker
from bokeh.models import ColumnDataSource, CDSView, CustomJS
from bokeh.models import DataTable, TableColumn
from bokeh.models import Div
from bokeh.models import TabPanel, Tabs, Switch
from bokeh.models import AllIndices, GroupFilter, IndexFilter
from bokeh.models import Div, Button, Select, TextInput
from bokeh.palettes import interp_palette
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark
from bokeh.models import ColumnDataSource, CDSView, CustomJS, Styles
from bokeh.events import ButtonClick


from modules.constants import *

default_hex_scatter_args = dict(
    marker="hex",
    line_width=1,
    angle=90,
    angle_units='deg',
    hover_fill_alpha=1,
    hover_line_alpha=1,
    hover_line_width=4,
    nonselection_alpha=.25,
    selection_fill_alpha=1.0,
    selection_line_width=6,
)


def init_plot():
    plot = figure(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools='tap,reset,save',
        toolbar_location='above',
        match_aspect=True,
        aspect_scale=1,
        title="SOM Grid Map",
        outline_line_width=0,
        name="hexwrapper",
    )
    plot.toolbar.logo = None
    plot.axis.visible = False
    plot.grid.visible = False

    colorbar_plot = figure(
        width=60,
        toolbar_location=None,
        outline_line_width=0,
        name="colorbar_plot"
        )

    return plot, colorbar_plot


def get_cds(som, meta_df):
    # Construct CDS
    weights = som.get_weights()

    # weights = som.get_weights()

    m = weights.shape[0]
    n = weights.shape[1]
    p = weights.shape[-1]  # number of inputs


    # X,Y coords of som hexes
    xe, ye = som.get_euclidean_coordinates()
    # m, n = ye.shape            # shape of som matrix

    # wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    # hsize = PLOT_HEIGHT/((0.5*n)+0.25)
    # size = np.min([wsize, hsize])

    size = np.min([PLOT_HEIGHT/(n+1), PLOT_WIDTH/(m+1)])
    # size = 10
    
    xx = (xe.reshape(m*n) + 1) * size / np.sqrt(3)*2
    yy = ye.reshape(m*n) * size
    # xx              = xe.reshape(m*n) * size
    
    # p = weights.shape[-1]   # number of inputs


    # Color codes
    umatrix = som.distance_map(scaling=SOM_SCALING)
    umatrix_long = umatrix.reshape(m*n)
    
    bmu_counter = Counter(meta_df['bmu'])
    hitsmatrix = np.zeros((m,n))
    
    for i in bmu_counter:
        u = np.unravel_index(i,(m,n))
        hitsmatrix[u] = bmu_counter[i]

    hits_long       = hitsmatrix.reshape(m*n)
    hits_pct        = hitsmatrix.reshape(m*n) / hitsmatrix.max()

    winmap = {}
    for i in range(m*n):
        u = np.unravel_index(i,(m,n))
        winmap[u] = meta_df.loc[meta_df['bmu']==i].index.tolist()

    winmap_long     = { np.ravel_multi_index(i,(m,n)): winmap[i] for i in winmap}

    # winmap = som.win_map(vec_vals, return_indices=True)
    # winmap_long = {np.ravel_multi_index(i, (m, n)): winmap[i] for i in winmap}

    codebook_long = weights.reshape((m*n, p))
    codebook_tree = KDTree(codebook_long)

    coords = ["{}".format(np.unravel_index(i,weights.shape[:2])) for i in range(m*n) ]

    node_indices = list(range(m*n))

    radius_multiplier = .75
    graph_radius_multiplier = .125

    node_topics = np.empty((m,n), dtype=list)
    node_topics_pct = np.empty((m,n), dtype=list)

    for wm in winmap.keys():
        item_ids = winmap[wm]
        node_topics[wm] = meta_df.iloc[item_ids,:]['Topic'].tolist()

    node_topics_mode = np.zeros((m,n))-2
    for j in range(n):
        for i in range(m):
            counter1 = Counter(node_topics[(i,j)])
            mf1 = counter1.most_common()[0][0]
            node_topics_mode[(i,j)] = mf1
            node_topics_pct[(i,j)] = counter1.most_common()[0][1]/len(node_topics[(i,j)])


    node_topics_long = node_topics_mode.reshape((m*n))
    node_topics_long = node_topics_long.astype(int).astype(str)
    node_topics_pct_long = node_topics_pct.reshape((m*n))

    recency = meta_df[['bmu','recency']].groupby('bmu').mean().reset_index()


    cds = dict(
        x=xx,
        y=yy,
        umatrix_long=umatrix_long,

        hits_long=hits_long,
        hits_pct=hits_pct * size * radius_multiplier,
        hits_radius=hits_long,
        radius=hits_long * graph_radius_multiplier,

        index=node_indices,

        coords=coords,
        topics = node_topics_long,

        topics_pct = node_topics_pct_long,
        topics_pct_deg = node_topics_pct_long*360,

        recency = recency['recency']


    )
    

    return cds, m, n, p, codebook_long


def make_plot_umatrix(plot, main_source, umatrix_long, m, n, p, palette='Viridis256'):


    wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    hsize = PLOT_HEIGHT/((0.5*n)+0.25)

    umatrix_cmap = LinearColorMapper(
        palette=palette,
        low=min(umatrix_long),
        high=max(umatrix_long)
    )

    # size = np.min([PLOT_HEIGHT/(n), PLOT_WIDTH/(m)])
    size = np.min([hsize, wsize])

    um_hex = plot.scatter(
        source=main_source,
        size=size+1,
        alpha=1,
        fill_color={"field": "umatrix_long",
                    "transform": umatrix_cmap},
        line_color={"field": "umatrix_long",
                    "transform": umatrix_cmap},
        hover_line_color=CONTRAST_COLOR1,
        selection_line_color=CONTRAST_COLOR1,
        **default_hex_scatter_args
    )

    um_colorbar = ColorBar(
                            color_mapper=umatrix_cmap, 
                            location=(0,0),
                            width=10,
                            height=500,
                            ticker=BasicTicker(min_interval=.1)
                            )

    
    return plot, um_hex, um_colorbar


def make_plot_hits(plot, main_source):
    # print(main_source.data)
    hits_hex = plot.scatter(source=main_source,
                size='hits_pct',
                marker="hex", 
                line_join='miter',
                fill_color="#fafafa",
                line_width=0,
                angle=90, 
                angle_units='deg',
                alpha=1,
                nonselection_alpha=1,
                selection_alpha=1,
           
    )

    hits_style = """
    :host(.active) .bar{background-color:#c6e6d8;}
    :host(.active) .knob{background-color:#31b57a;}
    """
    hits_switch = Switch(active=True, 
    stylesheets=[hits_style]
    )

    hits_switch.js_on_change("active", 
                            CustomJS(
                                    args=dict(hits=hits_hex),
                                    code="""hits.visible = !hits.visible"""))


    return plot, hits_hex, hits_switch


def annotate_plot(plot, plot_renderers):

    tooltips = [
        ("# Documents", "@hits_long"),
        ("Avg Distance", "@umatrix_long"),
        ("Top Topic", "@topics"),
        ("Avg Recency", "@recency"),
    ]

    plot.add_tools(HoverTool(tooltips=tooltips,
                             renderers=plot_renderers,
                            #  styles={"margin":"1em"}
                             ))

    return plot

def make_plot_topics(plot, main_source, json_path, m, n, p):
    wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    hsize = PLOT_HEIGHT/((0.75*n)+0.25)
    size = np.min([hsize,wsize])


    with open(json_path) as data:
        file_contents = data.read()
    
    topic_labels = json.loads(file_contents)

    topic_list = [str(i) for i in list(topic_labels.keys())]

    bg_hex = plot.scatter(source=main_source,
                size=size,
                color=factor_cmap('topics',  
                                palette=cc.palette['glasbey_category10'],
                                factors=topic_list),
                line_join='miter',
                alpha=.9,
                visible=False,
                hover_line_color="#DDDDDD",
                **default_hex_scatter_args
    )

    bg_wedge = plot.wedge(source=main_source,
                radius=size*.4,
                end_angle=0,
                start_angle='topics_pct_deg',
                direction='clock',
                color="white",
                line_color="black",
                line_join='miter',
                start_angle_units='deg',
                end_angle_units='deg',
                alpha=.45,
                visible=False,
                        
    )

    wedge_style = """
    :host(.active) .bar{background-color:#c6e6d8;}
    :host(.active) .knob{background-color:#31b57a;}
    """
    wedge_switch = Switch(active=True, name="wedge_switch", stylesheets=[wedge_style])
    wedge_switch.js_on_change("active", 
                            CustomJS(
                                    args=dict(w=bg_wedge),
                                    code="""w.visible = !w.visible"""))


    return plot, bg_hex, bg_wedge, topic_labels, wedge_switch

def make_plot_topoverlay(plot, vec_df, m, n, topic_labels, main_source, cds_dict, cb):

    wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    hsize = PLOT_HEIGHT/((0.75*n)+0.25)
    size = np.min([hsize,wsize])
    
    meta_topic_bmu = vec_df.loc[:,['Topic','bmu']].copy()
    tb_df = pd.crosstab(meta_topic_bmu.bmu,meta_topic_bmu.Topic)

    tb_df_long = tb_df.melt(ignore_index=False).reset_index()
    tb_df_long = tb_df_long[tb_df_long['value']>0].copy().reset_index(drop=True)
    tb_df_long.sort_values(by='value', ascending=False, inplace=True)
    tb_df_long.reset_index(drop=True, inplace=True)
    tb_df_long['pct'] = tb_df_long.apply(lambda row: np.max([.1,(row['value']/cds_dict['hits_long'][row['bmu']])]), axis=1 )
    tb_df_long['x'] = tb_df_long['bmu'].apply(lambda b: cds_dict['x'][b])
    tb_df_long['y'] = tb_df_long['bmu'].apply(lambda b: cds_dict['y'][b])
    t0_idx = tb_df_long.loc[tb_df_long['Topic']=='0'].index
    t0_view = CDSView(filter=IndexFilter(indices=t0_idx))
    to_source = ColumnDataSource(tb_df_long)

    topic_input_label = [(t, t + " - " + " ".join(topic_labels[t].split("_")[1:])) for t in topic_labels]


    toc1 = "#90DFD6"
    toc2 = "#86615C"
    toc3 = "#DFF6F4"

    to_bg_hex = plot.scatter(source=main_source,
                size=size,
                color=toc1,
                line_join='miter',
                line_color=toc1,
                line_alpha=.75,
                alpha=.85,
                visible=False,
                hover_line_color=toc2,
                **default_hex_scatter_args

    )

    to_pct_hex = plot.scatter(source=to_source,
                view=t0_view,
                size=size,
                color=toc2,
                line_join='miter',
                line_color=toc2,
                hover_line_color=toc2,
                alpha='pct',
                visible=False,
                **default_hex_scatter_args

    )

    to_dropdown = Select(value='x', 
                    options = [("x", "Select Topic")] + topic_input_label, 
                    visible=False,
                    max_width=200,
                    sizing_mode='stretch_width',
                    styles=dict({'min-width':'120px'}),
                    name='to_dropdown'
                    )
                    

    to_dropdown.js_on_change("value", 
                                CustomJS(args=dict( to_src = to_source,
                                                    to_view = t0_view,
                                                    to_index = IndexFilter(indices=[])
                                                    ), 
                                        code=cb['topic_dropdown']))

    return plot, to_bg_hex, to_pct_hex, to_dropdown


def make_recency_plot(plot, m,n, main_source, cds_dict):
    wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    hsize = PLOT_HEIGHT/((0.75*n)+0.25)
    size = np.min([hsize,wsize])

    rc_colors = ["#cb4663", "#fffeff", "#417a88"]

    rc_palette = interp_palette(rc_colors, 256)
    recency_cmap = LinearColorMapper(palette=rc_palette,
                            low = min(cds_dict['recency']), 
                            high = max(cds_dict['recency']))



    recency_hex = plot.scatter(source=main_source,
                size=size,
                fill_color={"field": "recency",
                        "transform": recency_cmap},
                line_join='miter',
                line_color="#999999",
                hover_line_color=rc_colors[2],
                visible=False,
                **default_hex_scatter_args

    )

    rc_colorbar = ColorBar(
                        color_mapper=recency_cmap, 
                        location=(0,0),
                        width=10,
                        height=500,
                        display_high=1.,
                        display_low=0.,
                        visible=False
                        )

    return plot, recency_hex, rc_colorbar



def make_query_hex(plot, m, n, x2, y2):
    wsize = (PLOT_WIDTH/(m+1))/(np.sqrt(3)/2)
    hsize = PLOT_HEIGHT/((0.75*n)+0.25)
    size = np.min([hsize,wsize])

    highlight_nodes = np.arange(m*n)
    highlight_rank = np.arange(m*n)

    highlight_rank = np.zeros(m*n)


    highlight_src = ColumnDataSource(dict(
            x=x2,
            y=y2,
            rank=highlight_rank,
            highlight=highlight_nodes
        )
    )
    highlight_view = CDSView(filter=IndexFilter(indices=[]))

    # highlight hex plot
    highlight_hex = plot.scatter(source=highlight_src,
                        size=size,
                        marker="hex", 
                        line_join='miter',
                        line_color="#FFF",
                        selection_line_alpha='rank',
                        nonselection_line_alpha='rank',                        
                        line_width=4,
                        fill_color=None,
                        angle=90, 
                        angle_units='deg',
                        nonselection_alpha='rank',
                        alpha='rank',
                        view=highlight_view
    )



    
    return plot, highlight_src, highlight_hex

def make_plots(vec_df, som, json_path, help_, cb):

    plot, colorbar_plot = init_plot()

    cds_dict, m, n, p, codebook_long = get_cds(som, vec_df)
    main_source = ColumnDataSource(cds_dict)
    
    ## TODO: Test  
    # range_padding 

    plot, um_hex, ub_cb = make_plot_umatrix(plot,
                                           main_source, cds_dict['umatrix_long'],
                                           m, n, p)

    plot, tp_hex, tp_wedge, topic_labels, wedge_switch = make_plot_topics(plot,
                                           main_source, json_path,
                                           m, n, p)

    
    colorbar_plot.add_layout(ub_cb, 'right')



    plot, to_bg_hex, to_pct_hex, to_dropdown = make_plot_topoverlay(plot, vec_df, m, n, 
                                                topic_labels, 
                                                main_source, cds_dict, cb)
    




    plot, recency_hex, rc_colorbar = make_recency_plot(plot, m, n, main_source, cds_dict)

    colorbar_plot.add_layout(rc_colorbar, 'right')

    plot, highlight_src, highlight_hex = make_query_hex(plot, m, n, cds_dict['x'], cds_dict['y'])




    plot, hits_hex, hits_switch = make_plot_hits(plot, main_source)
    hex_toggle_switch = row(hits_switch, Div(text="Toggle Hits Hex"))
    wedge_toggle_switch = row(wedge_switch, Div(text="Toggle Wedge"), 
                                visible=False,
                                name='wedge_toggle_switch')



    plot_renderers = [um_hex, 
                        tp_hex,
                        to_bg_hex, 
                        recency_hex
                        ]

    plot = annotate_plot(plot, plot_renderers)



    hex_dropdown = Select(value="um", 
                            options = [
                                ('um','UMatrix'), 
                                ('tp','Topic Model'), 
                                ('to',"Topic Overlay"),
                                ('rc',"Temporal Overlay"),
                                ], 
                            styles=dict({'min-width':'120px',
                            }),
                            max_width=200,
                            sizing_mode='stretch_width',

                            name="hex_dropdown"
                            )


    hex_dropdown.js_on_change('value',
            CustomJS(args=dict(um=um_hex,
                                um_cb=ub_cb, 
                                tp=tp_hex,
                                tp_wedge=tp_wedge,
                                w_switch=wedge_toggle_switch,
                                tb=to_bg_hex,
                                to=to_pct_hex,
                                ts=to_dropdown,
                                rc=recency_hex,
                                rc_cb=rc_colorbar,
                                hc=help_['contents'],
                                hb=help_['body'],
                                ht=help_['title']
                            ),
                    code=cb['hex_dropdown']
            )
        )


    plots_ = {'plot': plot, 'colorbar_plot': colorbar_plot}
    switches_ = {
        'hex_toggle_switch': hex_toggle_switch,
        'wedge_toggle_switch': wedge_toggle_switch
    }

    dropdowns_ = {
        'hex_dropdown': hex_dropdown,
        'to_dropdown': to_dropdown
    }

    highlights_ = {
        'highlight_src': highlight_src, 
        'highlight_hex': highlight_hex
    }


    return plots_, main_source, codebook_long, dropdowns_, switches_, highlights_






























def make_tables(vec_df, 
                meta_cols, table_cols_attrs, cb):

    placeholder_style = dict({
                          'padding': '1em',
                          'max-height': '150px',
                          'overflow': 'auto',
                          'box-shadow': 'rgba(0, 0, 0, 0.1) 0px 4px 6px -1px, rgba(0, 0, 0, 0.06) 0px 2px 4px -1px',
                          'border-radius': '.25em'
                          })

    table_cds = ColumnDataSource(vec_df[meta_cols])
    table_cols = [
        TableColumn(field=i,
                    title=i,
                    width=table_cols_attrs[i]['width']
                    ) for i in table_cols_attrs 
                    ]

    table_index = IndexFilter(indices=[])
    table_all   = AllIndices()
    table_none  = IndexFilter(indices=[])
    table_group = GroupFilter(column_name="bmu")

    table_view  = CDSView(filter=table_all)
    node_table = DataTable(source=table_cds, 
                        columns=table_cols,
                        view=table_view,
                        index_position=-1
                        )

    node_detail_text = "Click on  a row for details"
    node_detail = Div(text=node_detail_text, styles=placeholder_style)


    table_cds.selected.js_on_change('indices', 
                                    CustomJS(args=dict(
                                                table_src = table_cds,
                                                node_div = node_detail
                                                ),
                                            code=cb['row_click']
                                            ))

    ## Table attributes
    node_table.sizing_mode="stretch_width"
    node_table.width_policy="min"
    node_table.min_width=50
    node_table.min_height=100
    node_table.height_policy="max"
    node_table.max_width=PLOT_WIDTH
    node_table.max_height=PLOT_HEIGHT


    # node_detail.css_classes=["node_detail"]
    # node_detail.height = 150
    # node_detail.min_height = 100
    # node_detail.height_policy='min'
    # node_detail.max_width=PLOT_WIDTH

    # return node_detail, node_table, table_cds, table_view

    return node_table, table_cds, table_view, node_detail















def make_bars(vec_df, filter_col='user'):
    ## lists each bmu in which the user is present
    bar_df_ = vec_df.loc[:,[filter_col, 'bmu']].copy().reset_index(drop=True)

    ## creats a matrix of bmu x user 
    ## where each intersection contains how many times the user appears in that bmu
    bar_df = pd.crosstab(bar_df_.bmu, bar_df_[filter_col])
    ## stringify the bmu node index labels
    #bar_df.index = bar_df.index.map(str)

    ## transpose the bmu x user matrix and reset index to get a 'target' column
    bar_df_t = bar_df.T
    bar_df_t.reset_index(inplace=True)


    ## create a list of each combination of user and bmu and their intersections
    bar_df_long = bar_df.melt(ignore_index=False).reset_index()
    bar_df_long = bar_df_long[bar_df_long['value']>0].copy().reset_index(drop=True)
    ### SAME AS:
    #bar_df_['count'] = 1
    #bar_df_.groupby(['target','bmu']).count().reset_index().rename(columns={'count':'value'})


    bar_df_long.bmu = bar_df_long.bmu.astype(str)
    bar_df_long.sort_values(by='value', ascending=False, inplace=True)

    ## bar_df_long is a df containing each combination of bmu x user 
    ## and how many times this intersection occurs
    # bar_df_long

    ## Placeholders for index / view values and CDS filters
    index_0 = bar_df_long.loc[bar_df_long['bmu']=='0',:].index.to_list()
    bar_df_long_src = ColumnDataSource(bar_df_long)
    long_index = IndexFilter(indices=index_0)
    long_view = CDSView(filter=long_index)



    bar_df_idx = bar_df_long.loc[bar_df_long['bmu']=='0'].index.tolist()

    ## Get top 10 users by BR count to display bar chart when no node is selected
    top10_users_bar = bar_df_long.groupby(filter_col).sum(numeric_only=True).sort_values('value', ascending=False)[:10]
    top10_users_bar.reset_index(inplace=True)
    top10_users_bar['bmu'] = 'top'

    bar_df_long = pd.concat([bar_df_long,top10_users_bar], ignore_index=True)
    bar_df_long[filter_col] = bar_df_long[filter_col].astype(str)
    



    top10index = bar_df_long.loc[bar_df_long['bmu']=='top',:].index.to_list()
    
    bar_source = ColumnDataSource(data=bar_df_long)
    bar_index = IndexFilter(indices=top10index)
    bar_view = CDSView(filter=bar_index)

    x_range = bar_df_long.iloc[top10index][filter_col].tolist()
    

    ## Initialize bar plot
    bar_plot = figure(
        height=200,
        x_range=x_range,
        tooltips='Topic @Topic: @value documents',
        title="Top Topics"
        )

    bar_plot.toolbar.logo = None
    bar_plot.toolbar_location = None

    bars = bar_plot.vbar(x=filter_col, 
                source=bar_source, 
                width=.9, 
                view=bar_view,
                top='value',
                hover_fill_color="#bebebe", 
                fill_color="#959595",
                line_width= 0,
                )
    bar_plot.grid.visible = False
    bar_plot.axis.minor_tick_line_color = None
    bar_plot.xaxis.visible = False


    bar_plot.css_classes=["bar_plot"]
    bar_plot.sizing_mode="stretch_width"
    bar_plot.width_policy="min"
    bar_plot.min_width=50
    bar_plot.min_height=200
    bar_plot.height_policy="min"
    bar_plot.max_width=PLOT_WIDTH

    return bars, bar_plot, bar_view, bar_index, bar_source




def get_query_buttons(transformer, codebook_tree, highlights_, m, n):

    def calculate_query(event):

        topnn = codebook_tree.query(transformer.encode(query_input.value), TOPNN_K)[1].tolist()
        highlight_rank = np.zeros(m*n) #+ .5
        
        rank_alpha = (np.arange(TOPNN_K,0,-1)+1)/(1+TOPNN_K)
        for i in range(TOPNN_K):
            highlight_rank[topnn[i]] = np.max([rank_alpha[i],0.4])

        
        highlights_['highlight_src'].data['rank'] = highlight_rank
        highlights_['highlight_hex'].view = CDSView(filter=IndexFilter(indices=topnn))
        return

    def clear_query(event):
        query_input.value = ""
        highlights_['highlight_hex'].view = CDSView(filter=IndexFilter(indices=[]))


    run_query_button =Button(label="Run Query", button_type="success", styles=Styles(**BUTTON_STYLES))
    clear_query_button =Button(label="Clear Query", styles=Styles(**BUTTON_STYLES))


    run_query_button.on_event(ButtonClick, calculate_query)
    clear_query_button.on_event(ButtonClick, clear_query)

    query_input = TextInput(placeholder='Search Query', width=200)


    return run_query_button, clear_query_button, query_input


def get_help(hc, ht):

    help_body = Div(name="help_body", 
                    width=PLOT_WIDTH, 
                    width_policy='fixed', 
                    css_classes=["help_body"],
                    text=hc
    )

    help_title = Div(text=ht)


    help_col = column(
            help_title,
            help_body,
            name="help_text"
            )

    help_ = {
        'contents': hc,
        'body': help_body,
        'title': help_title
    }

    return help_, help_col