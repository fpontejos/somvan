import os
from scipy.spatial import KDTree


from bokeh.models import CustomJS
from bokeh.models import Div
from bokeh.models import TabPanel, Tabs
from bokeh.models import Styles
from bokeh.models import AllIndices, IndexFilter

from bokeh.layouts import row, column

from bokeh.io import curdoc

from modules.visualize import get_callbacks, get_help, get_query_buttons
from modules.visualize import make_plots, make_tables, make_bars

from modules.constants import DASH_CONFIG, HELP_BODY_CONTENTS, HELP_BODY_TITLE, PLOT_HEIGHT, PLOT_WIDTH
from modules.setup import get_meta, get_pickled_som, get_transformer
from modules.plot_components.graph import make_graph_plot, init_graph

ROOT_PATH = os.path.dirname(__file__)

CALLBACKS = get_callbacks(ROOT_PATH)


################################
# Setup
################################

som = get_pickled_som(DASH_CONFIG['data']['som'], ROOT_PATH)
mn, _ = som.get_euclidean_coordinates()
m, n = mn.shape            # shape of som matrix


help_, help_col = get_help(HELP_BODY_CONTENTS, HELP_BODY_TITLE)

meta_df = get_meta(DASH_CONFIG['data']['meta_csv'], ROOT_PATH)

json_path_ = [ROOT_PATH] + DASH_CONFIG['data']['topics'].split("/")
json_path = os.path.join(*json_path_)


################################
# Make main plots
################################

plots_, main_source, codebook_long, dropdowns_, switches_, highlights_ = make_plots(meta_df, 
                                    som, 
                                    json_path, 
                                    help_, 
                                    CALLBACKS)


################################
# Make tables
################################

node_table, table_source, table_view, node_detail = make_tables(
                        meta_df, 
                        DASH_CONFIG['details']['table']['meta_cols'], 
                        DASH_CONFIG['details']['table']['table_cols_attrs'],
                        CALLBACKS
                        )

################################
# Make bar plots
################################

bars, bar_plot, bar_view, bar_index, bar_source = make_bars(meta_df, 
                        filter_col='Topic')


################################
# Initialize graph
################################

G, Gpos, graph_plot, graph_cmap = init_graph(som, meta_df)

################################
# Make graph plot
################################

graph_plot, edge_source = make_graph_plot(G, Gpos, graph_plot, main_source, graph_cmap)


################################
# Add main callback
################################

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
                                            code=CALLBACKS['hex_click']
                                            ))




#######################
# Get query buttons
#######################


codebook_tree   = KDTree(codebook_long)
transformer = get_transformer(ROOT_PATH, DASH_CONFIG)


run_query_button, clear_query_button, query_input = get_query_buttons(transformer, codebook_tree, highlights_, m, n)


footer_input = column(
        Div(text="""<h3 style="margin-top:0px">Enter Query</h3>"""),
        query_input,
        row(run_query_button, 
            clear_query_button, 
                sizing_mode='stretch_width', 
                styles=Styles(justify_content="space-between")
            ),
        name="footer_input"

        )








#######################
# Arrange layout
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

curdoc().add_root(select_options)
curdoc().add_root(footer_input)
curdoc().add_root(help_col) 
curdoc().add_root(dropdowns_['hex_dropdown'])

curdoc().add_root(row(switches_['hex_toggle_switch'], name="hits_switch"))
curdoc().add_root(switches_['wedge_toggle_switch'])



curdoc().add_root(dropdowns_['to_dropdown']) 
curdoc().add_root(plots_['plot']) 
curdoc().add_root(plots_['colorbar_plot']) 
curdoc().add_root(details_tab_layout)  




curdoc().title = "Interactive SOM for Visual Analytics"
