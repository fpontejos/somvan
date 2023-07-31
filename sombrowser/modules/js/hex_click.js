
console.log('hex click callback')

var bmus = cb_obj.indices

// console.log(bmus)
/***
 * Trigger Edges selection on linked graph
 */
var coords = []

for(var i = 0; i < bmus.length; i++){
    coords.push(main_cds.data.coords[bmus[i]])
}
var coords_i = []
for(var i=0; i<coords.length; i++){
    var ci = coords[i]
    var ci_tmp = ci.slice(1,ci.length-1).replace(" ","").split(",")
    coords_i.push([parseInt(ci_tmp[0]),parseInt(ci_tmp[1])] )
}

var edge_idx_list = []

for(var i = 0; i < edge_src.get_length(); i++){
    if(bmus.includes(edge_src.data['start'][i]) || bmus.includes(edge_src.data['end'][i])){
        edge_idx_list.push(i)
    }
}

edge_src.selected.indices = edge_idx_list
edge_src.change.emit()

/***
 * End edges selection
 */

if (cb_obj.indices.length == 0) {
    // no hex selected
    console.log('no hex selected')
    table_view.filter = table_all
    


    var bar_indices = []
    var new_bar_range = []
    for (var i = 0; i < bar_source.get_length(); i++) {
        if (bar_source.data['bmu'][i] == 'top') {
            bar_indices.push(i)
            new_bar_range.push(bar_source.data['user'][i])
        }
    }

    
    
    bar_index.indices = bar_indices
    bar_view.filter = bar_index
    
    bar_plot.title.text = 'Top 10 Users'
    
    bar_plot.x_range.factors = new_bar_range;
    bar_source.change.emit()
    
} else {
    console.log('here', table_src.data)
    var indices = []
    var node_data = []
    console.log('bar_source', bar_source)

    var table_row = table_src.data
    console.log(bmus)
    console.log(table_src)

    for (var i=0; i<table_src.get_length(); i++) {

        if (bmus.includes(table_row.bmu[i])) {
        
            var table_row_data = {
                'meta.title': table_src.data['meta.title'][i],
                'meta.topic': table_src.data['meta.topic'][i],
                'content': table_src.data.content[i],
                'Topic': table_src.data.Topic[i]
            }

            node_data.push(table_row_data)
            indices.push(i)
        }
    }
    console.log(1)

    var bar_indices = []
    var new_bar_range = []
    
    for (var i = 0; i < bar_source.get_length(); i++) {
        if (bar_source.data['bmu'][i] == bmus[0].toString()) {
            bar_indices.push(i)
            new_bar_range.push(bar_source.data['Topic'][i])
        }
    }
    console.log(2)
    
    bar_index.indices = bar_indices
    bar_view.filter = bar_index
    
    var top_3 = (new_bar_range.slice(0,3))
    bar_plot.title.text = 'Top Topics: ' + top_3.join(', ')
    
    bar_plot.x_range.factors = new_bar_range;
    bar_source.change.emit()
    

    table_index.indices = indices
    table_view.filter = table_index
    
    
}

table_src.change.emit()
