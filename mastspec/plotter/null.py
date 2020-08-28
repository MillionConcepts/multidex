# callback creation section: register functions with app i/o

for axis in ['x', 'y']:
    app.callback(
        [
            Output('filter-1-'+axis, 'style'),
            Output('filter-2-'+axis, 'style'),
            Output('filter-3-'+axis, 'style'),
        ],
        [Input('axis-option-'+axis, 'value')]
    )(change_calc_input_visibility)

app.callback(
    Output('main-graph', 'figure'),
    # maybe later add an explicit recalc button?
    [
        *x_inputs, 
        *y_inputs, 
        Input({'type':'search-trigger', 'index':ALL}, 'value'), 
        Input('main-graph','hoverData')
    ],
    [
        State('main-graph','figure')
    ]
)(recalculate_graph)

app.callback(
    [
        Output({'type':'term-search', 'index':MATCH},'style'),
        Output({'type':'number-search-begin', 'index':MATCH},'style'),
        Output({'type':'number-search-end', 'index':MATCH},'style')
    ],
    [Input({'type':'field-search', 'index':MATCH},'value')],
    )(toggle_search_input_visibility)

app.callback(
    [Output({'type':'term-search', 'index':MATCH},'options'),
     Output({'type':'number-range-display','index':MATCH},'children')],
    [Input({'type':'field-search', 'index':MATCH},'value')],
    )(update_search_options)

app.callback(
        Output({'type':'search-trigger', 'index':0},'value'),
        [Input('submit-search', 'n_clicks')],
        [State({'type': 'field-search', 'index': ALL}, 'value'),
        State({'type': 'term-search', 'index': ALL}, 'value'),
        State({'type': 'number-search-begin', 'index': ALL}, 'value'),
        State({'type': 'number-search-end', 'index': ALL}, 'value')
        ],
    )(update_queryset)

app.callback(
    [
        Output('search-container', 'children'),
        Output({'type':'search-trigger', 'index':1},'value')
    ].
    [
        Input('add-param', 'n_clicks'), 
        Input({'type':'remove-param', "index":ALL},'n_clicks')
    ],
    [State('search-container', 'children')]
)(control_dropdowns)

# make graph viewer tabs

app.callback(
    [Output('tabs', 'children'), Output('tabs', 'value')],
    [Input('viewer-open-button','n_clicks'),
     Input({'type':'tab-close-button', "index":ALL},'n_clicks')
    ],
    [State('tabs','children')]
)(control_tabs)

# debug printer
# app.callback(
#     Output('garbage', 'children'),
#     [Input('tabs', 'value')]
# )(print_callback)


# right now main and view graph hover functions are basically duplicates, 
# but i'm reserving the possibility that they'll have different behaviors later

app.callback(
    Output({'type':'main-spec-image', 'index':0}, "children"), 
    [Input('main-graph', "hoverData")]
    )(update_spectrum_images)

app.callback(
    Output({'type':'main-spec-print', 'index':0}, "children"), 
    [Input('main-graph', "hoverData")]
    )(graph_point_to_metadata)

app.callback(
    Output({'type':'main-spec-graph','index':0},'figure'),
    [Input('main-graph','hoverData')]
)(update_spectrum_graph)

app.callback(
    Output({'type':'view-spec-image', 'index':MATCH}, "children"), 
    [Input({'type':'view-graph', "index":MATCH},'hoverData')]
    )(update_spectrum_images)

app.callback(
    Output({'type':'view-spec-print', 'index':MATCH}, "children"), 
    [Input({'type':'view-graph', "index":MATCH},'hoverData')]
    )(graph_point_to_metadata)

app.callback(
    Output({'type':'view-spec-graph','index':MATCH},'figure'),
    [Input({'type':'view-graph', "index":MATCH},'hoverData')]
)(update_spectrum_graph)

# def dummyfunc(data,figure):
#     hovered_point = data["points"][0]["customdata"]
#     figure.update_traces(
#        marker={'color':'red'}
#     )
#     return figure