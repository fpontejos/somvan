console.log('topic dropdown callback')

var selected_topic = cb_obj.value

var show_indices = []

for (var i = 0; i<to_src.get_length(); i++){
    if (to_src.data.Topic[i]==selected_topic) {
        show_indices.push(to_src.data.index[i])
    }
}

var idx = [...new Set(show_indices)]
to_index.indices = idx

to_view.filter = to_index
to_src.change.emit()