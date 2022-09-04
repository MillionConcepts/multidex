## frontend tests
* unit tests of component behavior
    * white-box unit tests of component initialization (does passing a mocked Dash component to React produce the expected HTML / SVG / js?)
    * white-box unit tests of component input (does passing a mocked browser event to an element instantiated from explicitly-defined HTML / SVG / js produce the expected DOM events, _specifically including a correctly-named Dash callback event_?)
* rendering / layout tests. for instance:
    * does passing mocked React-generated HTML/js/CSS result in a correctly-rendered page? 
    * does passing mocked user input to a component that does **not** produce a Dash callback result in the expected changes to the DOM? (integration tests for `condragulations.js` are a subset of this category)
    * do these things happen consistently at multiple zoom levels and window sizes?
* frontend asset-loading tests
    * do we find thumbnails when we look for them at the URL prefix corresponding to the specified Flask route? can we find the CSS and js files we expect?

## backend tests
* white-box unit tests for `condragulations.js`
* white-box unit tests of component creation (does passing a fixed set of inputs to a component factory function produce the expected Dash component object(s)?)

## frontend-backend tests
* layout tests -- for example, does passing a mocked Dash component to React produce the expected HTML/SVG/js? and then: does that code render correctly in the browser?

## import and export tests
* unit tests verifying correct ingest of individual file(s)
    * white-box -- for example: equality comparison between results of ingest ops on specific VISOR- or asdf-produced files and explicitly-defined data structures
    * black-box -- for example: pulling a random marslab file from Google Drive and verifying that ingest outcomes are "reasonable" by some metric
* integration tests verifying things like successful round-trip export & import (this requires nontrivial "success" metrics for application initialization)
