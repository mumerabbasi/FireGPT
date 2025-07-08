const map = L.map('map').setView([45.5236, -122.6750], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
}).addTo(map);

// Add search control
L.Control.geocoder({ defaultMarkGeocode: true }).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

// Icons for markers and controls (all L.icon!)
const fireIcon = L.icon({
  iconUrl: '/static/icons/fire.png',
  iconSize: [40, 40],
  iconAnchor: [20, 40],
  popupAnchor: [0, -40]
});
const droneIcon = L.icon({
  iconUrl: '/static/icons/drone.png',
  iconSize: [40, 40],
  iconAnchor: [20, 40],
  popupAnchor: [0, -40]
});
const fire_stationIcon = L.icon({
  iconUrl: '/static/icons/fire-station.png',
  iconSize: [40, 40],
  iconAnchor: [20, 40],
  popupAnchor: [0, -40]
});
const areaIcon = L.icon({
  iconUrl: '/static/icons/area.png',
  iconSize: [40, 40],
  iconAnchor: [20, 20],
  popupAnchor: [0, -20]
});
const clearIcon = L.icon({
  iconUrl: '/static/icons/clear.png',
  iconSize: [40, 40],
  iconAnchor: [20, 20],
  popupAnchor: [0, -20]
});
const clearAllIcon = L.icon({
  iconUrl: '/static/icons/clearall.png',
  iconSize: [40, 40],
  iconAnchor: [20, 20],
  popupAnchor: [0, -20]
});
const circleIcon = L.icon({
  iconUrl: '/static/icons/circle.png',
  iconSize: [16, 16],      // adjust size as needed
  iconAnchor: [8, 8],      // center the icon properly
  popupAnchor: [0, -8]
});

let selectedLayer = null;
let selectedWaypoints = [];

// Clear selection including polyline and linked waypoint circles
function clearSelection() {
  if (selectedLayer) {
    if (selectedLayer._icon) {
      selectedLayer._icon.classList.remove('selected-layer');
    } else if (selectedLayer.setStyle) {
      selectedLayer.setStyle({ color: '#353535' });
    }

    // Also clear highlight from associated waypoint circles
    if (selectedWaypoints.length > 0) {
      selectedWaypoints.forEach(circle => {
        if (circle._icon) {
          circle._icon.classList.remove('selected-layer');
        } else if (circle.setStyle) {
          circle.setStyle({ color: '#353535' });
        }
      });
      selectedWaypoints = [];
    }

    selectedLayer = null;
  }
}

// When selecting a layer
function layerSelection(e) {
  clearSelection();
  selectedLayer = e.layer;

  if (selectedLayer._icon) {
    // It's a marker or circle
    selectedLayer._icon.classList.add('selected-layer');
  } else if (selectedLayer.setStyle) {
    selectedLayer.setStyle({ color: 'red' });
  }

  // If this is a waypoint polyline, select its waypoint circles too
  if (selectedLayer._waypointCircles && Array.isArray(selectedLayer._waypointCircles)) {
    selectedWaypoints = selectedLayer._waypointCircles;
    selectedWaypoints.forEach(circle => {
      if (circle._icon) {
        circle._icon.classList.add('selected-layer');
      } else if (circle.setStyle) {
        circle.setStyle({ color: 'red' });
      }
    });
  }
}


// Clear selection when clicking on the map but not on any layer
map.on('click', function (e) {
  clearSelection();
});

drawnItems.on('click', function (e) {
  layerSelection(e);
});

const VerticalControl = L.Control.extend({
  onAdd: function (map) {
    const container = L.DomUtil.create('div', 'vertical-control leaflet-bar');

    // Controls array uses L.icon for all
    const controls = [
      { name: 'Fire', icon: fireIcon, type: 'marker' },
      { name: 'Drone', icon: droneIcon, type: 'marker' },
      { name: 'Fire station', icon: fire_stationIcon, type: 'marker' },
      { name: 'Area', icon: areaIcon, type: 'area' },
      { name: 'Clear', icon: clearIcon, type: 'clear' },
      { name: 'Clear All', icon: clearAllIcon, type: 'clearAll' }
    ];

    let currentDrawHandler = null;
    const ids = {
      Fire: 1,
      Drone: 1,
      'Fire station': 1,
      Area: 1
    };

    function cancelDrawing() {
      if (currentDrawHandler) {
        currentDrawHandler.disable();
        currentDrawHandler = null;
      }
    }

    function addCancelOnEscape() {
      function onKeyDown(e) {
        if (e.key === 'Escape') cancelDrawing();
      }
      document.addEventListener('keydown', onKeyDown, { once: true });
    }

    controls.forEach(({ name, icon, type }) => {
      const item = L.DomUtil.create('div', 'control-item', container);
      const btn = L.DomUtil.create('button', '', item);
      btn.title = name;

      // Render icon using L.icon.options.iconUrl
      const img = document.createElement('img');
      img.src = icon.options.iconUrl;
      btn.appendChild(img);

      btn.onclick = (e) => {
        e.stopPropagation();
        cancelDrawing();

        if (type === 'marker') {
          clearSelection();
          currentDrawHandler = new L.Draw.Marker(map, { icon });
          currentDrawHandler.enable();

          map.once(L.Draw.Event.CREATED, function (e) {
            const layer = e.layer;
            drawnItems.addLayer(layer);

            // need to be extacted
            layer.extract = true;

            const count = ids[name]++;
            const label = `${name.toUpperCase()} ${count}`;

            let popupLatLng;
            if (layer.getLatLng) {
              popupLatLng = layer.getLatLng();
            } else if (layer.getBounds) {
              popupLatLng = layer.getBounds().getCenter();
            }

            const coordInfo = layer.getLatLng
              ? `Lat: ${popupLatLng.lat.toFixed(5)}, Lng: ${popupLatLng.lng.toFixed(5)}`
              : (() => {
                const bounds = layer.getBounds();
                const nw = bounds.getNorthWest();
                const se = bounds.getSouthEast();
                return `
                    <b>NW:</b> (${nw.lat.toFixed(5)}, ${nw.lng.toFixed(5)})<br>
                    <b>SE:</b> (${se.lat.toFixed(5)}, ${se.lng.toFixed(5)})
                  `;
              })();

            const defaultPopupContent = `${label}<br>${coordInfo}`;
            const formHtml = `
              <form id="desc-form" style="min-width: 200px;">
                <label for="desc-input"><b>${label}</b> Description (optional):</label><br>
                <textarea id="desc-input" rows="3" style="width: 100%; resize: vertical;"></textarea><br>
                <small>${coordInfo}</small><br>
                <button type="button" id="cancel-btn" style="margin-left: 6px;">Cancel</button>
              </form>
            `;

            const popup = L.popup({ closeOnClick: false, autoClose: false })
              .setLatLng(popupLatLng)
              .setContent(formHtml)
              .openOn(map);

            function closePopup() {
              map.closePopup(popup);
              document.removeEventListener('click', onDocumentClick, true);
            }

            const form = document.getElementById('desc-form');
            const cancelBtn = document.getElementById('cancel-btn');
            const textarea = document.getElementById('desc-input');
            textarea.focus();

            form.addEventListener('submit', (evt) => {
              evt.preventDefault();
              const desc = textarea.value.trim();

              const fullLabel = desc
                ? `${label}: ${desc}<br>${coordInfo}`
                : defaultPopupContent;

              layer.bindPopup(fullLabel);
              closePopup();
            });

            cancelBtn.addEventListener('click', () => {
              layer.bindPopup(defaultPopupContent);
              closePopup();
            });

            textarea.addEventListener('keydown', (evt) => {
              if (evt.key === 'Enter' && !evt.shiftKey) {
                evt.preventDefault();
                form.requestSubmit();
              }
            });

            function onDocumentClick(e) {
              const popupContainer = document.querySelector('.leaflet-popup-content');
              if (popupContainer && !popupContainer.contains(e.target)) {
                layer.bindPopup(defaultPopupContent);
                closePopup();
              }
            }
            document.addEventListener('click', onDocumentClick, true);

            currentDrawHandler = null;
          });

          addCancelOnEscape();
        }
        else if (type === 'area') {
          clearSelection();
          currentDrawHandler = new L.Draw.Rectangle(map, {
            shapeOptions: {
              color: '#353535',
              fillColor: '#353535',
              fillOpacity: 0.4  // Optional: control fill transparency
            }
          });
          currentDrawHandler.enable();

          map.once(L.Draw.Event.CREATED, function (e) {
            const layer = e.layer;
            drawnItems.addLayer(layer);

            // need to be extacted
            layer.extract = true;

            const count = ids[name]++;
            const label = `${name.toUpperCase()} ${count}`;

            let popupLatLng;
            if (layer.getLatLng) {
              popupLatLng = layer.getLatLng();
            } else if (layer.getBounds) {
              popupLatLng = layer.getBounds().getCenter();
            }

            const coordInfo = layer.getLatLng
              ? `Lat: ${popupLatLng.lat.toFixed(5)}, Lng: ${popupLatLng.lng.toFixed(5)}`
              : (() => {
                const bounds = layer.getBounds();
                const nw = bounds.getNorthWest();
                const se = bounds.getSouthEast();
                return `
                    <b>NW:</b> (${nw.lat.toFixed(5)}, ${nw.lng.toFixed(5)})<br>
                    <b>SE:</b> (${se.lat.toFixed(5)}, ${se.lng.toFixed(5)})
                  `;
              })();

            const defaultPopupContent = `${label}<br>${coordInfo}`;
            const formHtml = `
              <form id="desc-form" style="min-width: 200px;">
                <label for="desc-input"><b>${label}</b> Description (optional):</label><br>
                <textarea id="desc-input" rows="3" style="width: 100%; resize: vertical;"></textarea><br>
                <small>${coordInfo}</small><br>
                <button type="button" id="cancel-btn" style="margin-left: 6px;">Cancel</button>
              </form>
            `;

            const popup = L.popup({ closeOnClick: false, autoClose: false })
              .setLatLng(popupLatLng)
              .setContent(formHtml)
              .openOn(map);

            function closePopup() {
              map.closePopup(popup);
              document.removeEventListener('click', onDocumentClick, true);
            }

            const form = document.getElementById('desc-form');
            const cancelBtn = document.getElementById('cancel-btn');
            const textarea = document.getElementById('desc-input');
            textarea.focus();

            form.addEventListener('submit', (evt) => {
              evt.preventDefault();
              const desc = textarea.value.trim();

              const fullLabel = desc
                ? `${label}: ${desc}<br>${coordInfo}`
                : defaultPopupContent;

              layer.bindPopup(fullLabel);
              closePopup();
            });

            cancelBtn.addEventListener('click', () => {
              layer.bindPopup(defaultPopupContent);
              closePopup();
            });

            textarea.addEventListener('keydown', (evt) => {
              if (evt.key === 'Enter' && !evt.shiftKey) {
                evt.preventDefault();
                form.requestSubmit();
              }
            });

            function onDocumentClick(e) {
              const popupContainer = document.querySelector('.leaflet-popup-content');
              if (popupContainer && !popupContainer.contains(e.target)) {
                layer.bindPopup(defaultPopupContent);
                closePopup();
              }
            }
            document.addEventListener('click', onDocumentClick, true);

            currentDrawHandler = null;
          });

          addCancelOnEscape();
        }
        else if (type === 'clear') {
          if (selectedLayer) {
            drawnItems.removeLayer(selectedLayer);

            // Remove all the selected way points if any
            if (selectedWaypoints.length > 0) {
              selectedWaypoints.forEach(circle => { drawnItems.removeLayer(circle) });
              selectedWaypoints = [];
            }

            selectedLayer = null;
          } else {
            const center = map.getCenter();
            const noSelectionPopup = L.popup({
              closeButton: true,
              autoClose: true,
              closeOnClick: true,
              className: 'custom-popup'
            })
              .setLatLng(center)
              .setContent('<b>No shape selected to clear!</b>')
              .openOn(map);
          }
        }
        else if (type === 'clearAll') {
          drawnItems.clearLayers();
          selectedLayer = null;
          Object.keys(ids).forEach(k => ids[k] = 1);
        }
      };

      // Add label under button
      const label = L.DomUtil.create('span', 'control-label', item);
      label.innerText = name;
    });

    return container;
  }
});

const customControl = new VerticalControl();
map.whenReady(() => {
  customControl.addTo(map);
  const controlEl = document.querySelector('.vertical-control');
  if (controlEl) {
    document.getElementById('map').appendChild(controlEl);
  }
});

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
function extract_map_features_post() {
  const marker = [];
  const rectangle = [];

  drawnItems.eachLayer(layer => {

    let name = '';
    let id = 0;
    let description = '';
    let coordinates = null;

    const popupContent = layer.getPopup() ? layer.getPopup().getContent() : '';
    const nameMatch = popupContent.match(/([A-Z ]+)(\d+)/);
    if (nameMatch) {
      name = nameMatch[1].trim();
      id = parseInt(nameMatch[2]);
    }

    const descMatch = popupContent.match(/:\s*(.*)<br>/);
    description = descMatch ? descMatch[1].trim() : '';

    if (layer instanceof L.Marker) {

      if (layer.extract) {
        const latlng = layer.getLatLng();
        coordinates = { lat: latlng.lat, lng: latlng.lng };

        marker.push({ name, id, description, coordinates });
      }

    } else if (layer instanceof L.Rectangle) {
      // Special case for Area 
      const descMatch = popupContent.match(/^[^:<]+:\s*(.*?)<br>/);
      description = descMatch ? descMatch[1].trim() : '';

      const bounds = layer.getBounds();
      const nw = bounds.getNorthWest();
      const se = bounds.getSouthEast();

      coordinates = {
        nw: { lat: nw.lat, lng: nw.lng },
        se: { lat: se.lat, lng: se.lng }
      };

      rectangle.push({ name, id, description, coordinates });
    }
  });

  return { rectangle, marker }
}
//////////////////////////////////////////////////////////////
function drawWaypointPath(coordsArray) {
  if (!Array.isArray(coordsArray) || coordsArray.length < 2) {
    console.error("Need at least two coordinates to draw a waypoint path.");
    return;
  }

  const latlngs = coordsArray.map(coord => {
    if (Array.isArray(coord) && coord.length === 2) {
      return L.latLng(coord[0], coord[1]);
    } else if (coord.lat !== undefined && coord.lng !== undefined) {
      return L.latLng(coord.lat, coord.lng);
    } else {
      console.error("Invalid coordinate format:", coord);
      return null;
    }
  }).filter(Boolean);

  // Draw polyline
  const polyline = L.polyline(latlngs, {
    color: '#353535',
    weight: 4,
    opacity: 0.8,
    lineJoin: 'round',
    dashArray: '6, 4'
  }).addTo(drawnItems);

  // Draw waypoint circles and keep references
  const waypointCircles = latlngs.map(latlng => {
    const circle = L.marker(latlng, { icon: circleIcon }).addTo(drawnItems);
    return circle;
  });

  // helper variable to skip this in extraction of features
  waypointCircles.extract = false;

  // Link circles with polyline for selection
  polyline._waypointCircles = waypointCircles;
  polyline.extract = false;

  // Zoom to fit the full route
  map.fitBounds(polyline.getBounds());

  // Add popup summary
  polyline.bindPopup(`<b>WAYPOINT ROUTE</b><br>Points: ${latlngs.length}`).openPopup();
}

//////////////////////////////////////////////////////////////////////////
