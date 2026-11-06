// Writes `maps.json`, the map frames the pipeline needs, from the maps of SquadCalc (src/data/maps.js):
//
//     node scripts/export_maps_json.mjs <path to SquadCalc>/src/data/maps.js > maps.json
//
// {"<slug>": {"min": [x, y], "size": [width, height]}} in metres: the north-west corner of the map image and its size, the
// frame the Detailed tiles and the whole-map texture are stretched over.
const { MAPS, initMapsProperties } = await import(process.argv[2]);
initMapsProperties();
const frames = {};
for (const map of MAPS) {
    const { corner0, corner1 } = map.SDK_data.minimap;
    frames[map.name.toLowerCase()] = {
        min: [Math.min(corner0[0], corner1[0]), Math.min(corner0[1], corner1[1])],
        size: [map.size, map.sizeY],
    };
}
console.log(JSON.stringify(frames));
