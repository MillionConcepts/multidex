// Make the DIV element draggable:

let whatAreWeDragging = null;
let mouseDragTracker = {'x': 0, 'y': 0};
let currentEvent = null;
let draggables = [];
let visibilityRecord = {};

const drag = function (event) {
    event.preventDefault();
    const xMotion = mouseDragTracker['x'] - event.clientX;
    const yMotion = mouseDragTracker['y'] - event.clientY;
    mouseDragTracker['x'] = event.clientX;
    mouseDragTracker['y'] = event.clientY;
    whatAreWeDragging.style.top = (whatAreWeDragging.offsetTop - yMotion) + "px";
    whatAreWeDragging.style.left = (whatAreWeDragging.offsetLeft - xMotion) + "px";
}

const unTuck = function () {
    document.onmouseup = null;
    document.onmousemove = null;
    whatAreWeDragging = null;
}

const getIntoDrag = function (event) {
    event.preventDefault();
    event.path.forEach(
        function isItDraggable(element) {
            if (draggables.includes(element.id)) {
                whatAreWeDragging = element;
            }
        }
    );
    mouseDragTracker['x'] = event.clientX;
    mouseDragTracker['y'] = event.clientY;
    document.onmouseup = unTuck;
    document.onmousemove = drag;
};

const makeDraggable = function (handleElementId, targetElementId) {
    draggables.push(targetElementId)
    document.getElementById(handleElementId).onmousedown = getIntoDrag
    // document.getElementById(elementId).ondblclick = toggleSize
};

const visToggler = function(elementId) {
    return function() {
        let element = document.getElementById(elementId);
        if (element.style.display !== 'none') {
            visibilityRecord[elementId] = element.style.display;
            element.style.display = 'none';
        }
        else {
            if (Object.keys(visibilityRecord).includes(elementId)) {
                element.style.display = visibilityRecord[elementId];
            }
            else {
                element.style.display = 'revert';
            }
        }
    }
};

const makeHider = function (handleElementId, targetElementId) {
    document.getElementById(handleElementId).ondblclick = visToggler(targetElementId);
};