console.log('hex dropdown callback')

// make everything invisible
um.visible = false
um_cb.visible = false

tp.visible = false
tp_wedge.visible = false
w_switch.visible = false

tb.visible = false
to.visible = false
ts.visible = false

rc.visible = false
rc_cb.visible = false

// show selected
var selected_hex = cb_obj.value

if (selected_hex == 'um') {
    um.visible = true
    um_cb.visible = true
    ht.text = "<h3 style='margin-top:0px'>" + hc.um['title'] + "</h3>"
    hb.text = "<p style='margin-bottom:1em'>" + hc.um['body'].join("</p><p style='margin-bottom:1em'>") + "</p>"
} else if (selected_hex == 'tp') {
    tp.visible = true
    tp_wedge.visible = true
    w_switch.visible = true
    ht.text = "<h3 style='margin-top:0px'>" + hc.tp['title'] + "</h3>"
    hb.text = "<p style='margin-bottom:1em'>" + hc.tp['body'].join("</p><p style='margin-bottom:1em'>") + "</p>"

} else if (selected_hex == 'to') {
    tb.visible = true
    to.visible = true
    ts.visible = true
    ht.text = "<h3 style='margin-top:0px'>" + hc.to['title'] + "</h3>"
    hb.text = "<p style='margin-bottom:1em'>" + hc.to['body'].join("</p><p style='margin-bottom:1em'>") + "</p>"
} else if (selected_hex == 'rc') {
    rc.visible = true
    rc_cb.visible = true
    ht.text = "<h3 style='margin-top:0px'>" + hc.rc['title'] + "</h3>"
    hb.text = "<p style='margin-bottom:1em'>" + hc.rc['body'].join("</p><p>") + "</p>"
}

