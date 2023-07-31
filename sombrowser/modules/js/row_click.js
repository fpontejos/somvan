var row_idx = cb_obj.indices[0]
var content = table_src.data.content[row_idx].split('\\n').join('<p class="detail_p">')

var data_ = {
'bmu': table_src.data.bmu[row_idx],
'topic': table_src.data['topic_name'][row_idx],
'title': table_src.data['meta.title'][row_idx],
'url': table_src.data['meta.link'][row_idx]
}

let topics = data_['topic'].replaceAll(" ", ", ")

node_div.text = `
<div style="padding: .5em .5em .5em .25em">
<strong>${data_['title']}</strong>
<br>
<em>${topics}</em>

<br>
<br>
${content}

<br>
<br>

<a href="${data_['url']}">${data_['url']}</a>
</div>
`