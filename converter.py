"""
ASV Navigation System - Boundary Converter
Converts GeoJSON / GPX / KML / raw OSM → the canonical JSON format
that map_handler.py and the navigation system consume.

Canonical output schema (mission_boundary.json):
{
  "mission":        "<string>",
  "ground_station": {"lat": float, "lon": float},
  "boundary":       [{"lat": float, "lon": float}, ...]
}

Usage (CLI):
    python converter.py input.geojson -o mission_boundary.json
    python converter.py input.gpx     -o mission_boundary.json
    python converter.py input.kml     -o mission_boundary.json
    python converter.py input.osm     -o mission_boundary.json --osm-relation 12345
    python converter.py input.osm     -o mission_boundary.json --osm-way 67890

Usage (API):
    from converter import BoundaryConverter
    coords = BoundaryConverter.from_geojson("lake.geojson")
    BoundaryConverter.save(coords, "mission_boundary.json")
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

log = logging.getLogger("Converter")

# Type alias
LatLon = Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Core converter class
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryConverter:
    """
    Static factory: parse any supported format → list of (lat, lon) tuples.
    Then call save() to write the canonical JSON mission file.
    """

    # ── Public dispatch ───────────────────────────────────────────────────────

    @staticmethod
    def load(path: str, **kwargs) -> List[LatLon]:
        """
        Auto-detect format from file extension and parse.

        Supported extensions:
            .geojson  → GeoJSON
            .json     → GeoJSON (assumed)
            .gpx      → GPX
            .kml      → KML
            .osm      → OpenStreetMap XML (pass osm_way=<id> or osm_relation=<id>)
        """
        p = Path(path)
        ext = p.suffix.lower()

        if ext in (".geojson", ".json"):
            return BoundaryConverter.from_geojson(path)
        elif ext == ".gpx":
            return BoundaryConverter.from_gpx(path)
        elif ext == ".kml":
            return BoundaryConverter.from_kml(path)
        elif ext == ".osm":
            way_id      = kwargs.get("osm_way")
            relation_id = kwargs.get("osm_relation")
            return BoundaryConverter.from_osm(path, way_id=way_id, relation_id=relation_id)
        else:
            raise ValueError(
                f"Unknown extension '{ext}'. "
                "Supported: .geojson, .json, .gpx, .kml, .osm"
            )

    @staticmethod
    def save(coords: List[LatLon], output_path: str,
             mission_name: str = "asv_mission",
             ground_station: Optional[LatLon] = None) -> None:
        """
        Write canonical mission JSON.

        Args:
            coords:         Ordered (lat, lon) boundary polygon.
            output_path:    Destination file path.
            mission_name:   Human-readable mission identifier.
            ground_station: Optional (lat, lon) of launch point.
                            Defaults to the first polygon vertex.
        """
        if not coords:
            raise ValueError("Cannot save empty coordinate list.")

        gs = ground_station or coords[0]

        # Ensure polygon is closed in output
        boundary = list(coords)
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])

        data = {
            "mission":        mission_name,
            "description":    f"Auto-converted boundary ({len(boundary)-1} vertices)",
            "ground_station": {"lat": gs[0], "lon": gs[1]},
            "boundary": [{"lat": lat, "lon": lon} for lat, lon in boundary],
            "config": {
                "sweep_width_m":   3.0,
                "sweep_angle_deg": 0,
                "waypoint_radius_m": 2.0,
                "cruise_speed": 0.6
            }
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved %d-vertex boundary → %s", len(boundary) - 1, output_path)

    # ── GeoJSON ───────────────────────────────────────────────────────────────

    @staticmethod
    def from_geojson(path: str) -> List[LatLon]:
        """
        Parse a GeoJSON file containing a Polygon or MultiPolygon feature.

        GeoJSON coordinate order is [longitude, latitude] (note: reversed vs GPS).
        We extract the outer ring of the first polygon found.

        Handles:
            • FeatureCollection → first Polygon/MultiPolygon feature
            • Feature           → geometry directly
            • Geometry          → Polygon or MultiPolygon directly
        """
        with open(path) as f:
            data = json.load(f)

        geo = BoundaryConverter._geojson_extract_geometry(data)
        if geo is None:
            raise ValueError("No Polygon or MultiPolygon geometry found in GeoJSON.")

        if geo["type"] == "Polygon":
            # Outer ring is index 0; holes are 1..n
            ring = geo["coordinates"][0]
        elif geo["type"] == "MultiPolygon":
            # Take the largest polygon (most coordinates)
            rings = [poly[0] for poly in geo["coordinates"]]
            ring  = max(rings, key=len)
        else:
            raise ValueError(f"Expected Polygon/MultiPolygon, got {geo['type']}")

        # GeoJSON: [lon, lat, ?elevation] → (lat, lon)
        coords = [(pt[1], pt[0]) for pt in ring]
        BoundaryConverter._validate(coords, "GeoJSON")
        return coords

    @staticmethod
    def _geojson_extract_geometry(data: dict) -> Optional[dict]:
        t = data.get("type")
        if t == "FeatureCollection":
            for feat in data.get("features", []):
                g = feat.get("geometry", {})
                if g.get("type") in ("Polygon", "MultiPolygon"):
                    return g
        elif t == "Feature":
            return data.get("geometry")
        elif t in ("Polygon", "MultiPolygon"):
            return data
        return None

    # ── GPX ──────────────────────────────────────────────────────────────────

    @staticmethod
    def from_gpx(path: str) -> List[LatLon]:
        """
        Parse a GPX file.

        Extracts boundary coordinates from (in priority order):
          1. <trk>/<trkseg>/<trkpt>  — track points (most common for surveys)
          2. <rte>/<rtept>           — route points
          3. <wpt>                   — standalone waypoints

        GPX uses lat/lon attributes: lat="12.34" lon="77.56"
        """
        tree = ET.parse(path)
        root = tree.getroot()

        # Handle XML namespace (GPX 1.1 uses xmlns="http://www.topografix.com/GPX/1/1")
        ns = BoundaryConverter._gpx_ns(root)

        # Try tracks first
        coords = BoundaryConverter._gpx_track_coords(root, ns)
        if not coords:
            coords = BoundaryConverter._gpx_route_coords(root, ns)
        if not coords:
            coords = BoundaryConverter._gpx_waypoint_coords(root, ns)

        if not coords:
            raise ValueError("No track, route, or waypoint coordinates found in GPX.")

        BoundaryConverter._validate(coords, "GPX")
        return coords

    @staticmethod
    def _gpx_ns(root: ET.Element) -> str:
        """Extract namespace string from root tag, e.g. '{http://...}'."""
        m = re.match(r"\{.*\}", root.tag)
        return m.group(0) if m else ""

    @staticmethod
    def _gpx_track_coords(root: ET.Element, ns: str) -> List[LatLon]:
        coords = []
        for trk in root.findall(f"{ns}trk"):
            for seg in trk.findall(f"{ns}trkseg"):
                for pt in seg.findall(f"{ns}trkpt"):
                    coords.append((float(pt.attrib["lat"]), float(pt.attrib["lon"])))
        return coords

    @staticmethod
    def _gpx_route_coords(root: ET.Element, ns: str) -> List[LatLon]:
        coords = []
        for rte in root.findall(f"{ns}rte"):
            for pt in rte.findall(f"{ns}rtept"):
                coords.append((float(pt.attrib["lat"]), float(pt.attrib["lon"])))
        return coords

    @staticmethod
    def _gpx_waypoint_coords(root: ET.Element, ns: str) -> List[LatLon]:
        return [
            (float(pt.attrib["lat"]), float(pt.attrib["lon"]))
            for pt in root.findall(f"{ns}wpt")
        ]

    # ── KML ──────────────────────────────────────────────────────────────────

    @staticmethod
    def from_kml(path: str) -> List[LatLon]:
        """
        Parse a KML file.

        Searches for <Polygon><outerBoundaryIs><LinearRing><coordinates>
        or falls back to <LineString><coordinates> / <Point> sequences.

        KML coordinate format: lon,lat[,elevation] space-separated.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        ns   = BoundaryConverter._kml_ns(root)

        # Try Polygon outer boundary first
        coords = BoundaryConverter._kml_polygon_coords(root, ns)
        if not coords:
            coords = BoundaryConverter._kml_linestring_coords(root, ns)
        if not coords:
            raise ValueError("No Polygon or LineString found in KML.")

        BoundaryConverter._validate(coords, "KML")
        return coords

    @staticmethod
    def _kml_ns(root: ET.Element) -> str:
        m = re.match(r"\{.*\}", root.tag)
        return m.group(0) if m else ""

    @staticmethod
    def _kml_parse_coord_string(text: str) -> List[LatLon]:
        """
        Parse KML coordinate string 'lon,lat,ele lon,lat,ele ...' → (lat,lon) list.
        Handles both space and newline separators.
        """
        coords = []
        for token in text.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lat, lon))
        return coords

    @staticmethod
    def _kml_polygon_coords(root: ET.Element, ns: str) -> List[LatLon]:
        for poly in root.iter(f"{ns}Polygon"):
            outer = poly.find(f".//{ns}outerBoundaryIs//{ns}coordinates")
            if outer is not None and outer.text:
                return BoundaryConverter._kml_parse_coord_string(outer.text)
        return []

    @staticmethod
    def _kml_linestring_coords(root: ET.Element, ns: str) -> List[LatLon]:
        for ls in root.iter(f"{ns}LineString"):
            coord_el = ls.find(f"{ns}coordinates")
            if coord_el is not None and coord_el.text:
                return BoundaryConverter._kml_parse_coord_string(coord_el.text)
        return []

    # ── OSM ──────────────────────────────────────────────────────────────────

    @staticmethod
    def from_osm(path: str,
                 way_id:      Optional[int] = None,
                 relation_id: Optional[int] = None) -> List[LatLon]:
        """
        Parse raw OSM XML.

        OSM XML structure:
            <osm>
              <node id="..." lat="..." lon="..."/>
              ...
              <way id="...">
                <nd ref="<node_id>"/>
                ...
              </way>
              <relation id="...">
                <member type="way" ref="<way_id>" role="outer"/>
                ...
              </relation>
            </osm>

        Strategy:
          • If relation_id given: find that relation → collect outer-role ways
            → stitch node sequences → return coordinate ring.
          • If way_id given: find that way → node sequence → coordinates.
          • If neither given: use the first closed way (where first node == last node).

        Tip: download OSM XML with Overpass:
            [out:xml];
            (way["natural"="water"](around:500,LAT,LON);>;);
            out body;
        """
        tree = ET.parse(path)
        root = tree.getroot()

        # Build node_id → (lat, lon) index
        nodes: dict[int, LatLon] = {}
        for node in root.findall("node"):
            nid = int(node.attrib["id"])
            lat = float(node.attrib["lat"])
            lon = float(node.attrib["lon"])
            nodes[nid] = (lat, lon)

        if not nodes:
            raise ValueError("No <node> elements found in OSM file.")

        # Build way_id → [node_id, ...] index
        ways: dict[int, List[int]] = {}
        for way in root.findall("way"):
            wid  = int(way.attrib["id"])
            refs = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
            ways[wid] = refs

        # ── Relation-based extraction ──────────────────────────────────────
        if relation_id is not None:
            return BoundaryConverter._osm_from_relation(
                root, nodes, ways, int(relation_id)
            )

        # ── Direct way extraction ──────────────────────────────────────────
        if way_id is not None:
            return BoundaryConverter._osm_way_to_coords(
                int(way_id), ways, nodes
            )

        # ── Auto-detect: first closed way ────────────────────────────────
        for wid, refs in ways.items():
            if len(refs) >= 4 and refs[0] == refs[-1]:
                log.info("OSM auto-detected closed way id=%d (%d nodes)", wid, len(refs))
                coords = [nodes[r] for r in refs if r in nodes]
                BoundaryConverter._validate(coords, "OSM")
                return coords

        raise ValueError(
            "Could not find a closed way in OSM file. "
            "Specify --osm-way <id> or --osm-relation <id>."
        )

    @staticmethod
    def _osm_way_to_coords(way_id: int, ways: dict, nodes: dict) -> List[LatLon]:
        refs = ways.get(way_id)
        if refs is None:
            raise ValueError(f"Way id={way_id} not found in OSM file.")
        coords = [nodes[r] for r in refs if r in nodes]
        if len(coords) < 3:
            raise ValueError(f"Way id={way_id} has < 3 resolvable nodes.")
        BoundaryConverter._validate(coords, f"OSM way {way_id}")
        return coords

    @staticmethod
    def _osm_from_relation(root: ET.Element,
                           nodes: dict, ways: dict,
                           relation_id: int) -> List[LatLon]:
        """
        Stitch together outer-role ways of a multipolygon relation.
        OSM ways may be stored in any order; we chain them by matching
        endpoint node IDs.
        """
        rel = None
        for r in root.findall("relation"):
            if int(r.attrib["id"]) == relation_id:
                rel = r
                break
        if rel is None:
            raise ValueError(f"Relation id={relation_id} not found in OSM file.")

        outer_way_ids = [
            int(m.attrib["ref"])
            for m in rel.findall("member")
            if m.attrib.get("type") == "way" and m.attrib.get("role") == "outer"
        ]
        if not outer_way_ids:
            raise ValueError(f"Relation {relation_id} has no outer-role way members.")

        # Chain ways into a single ring
        segments = [ways[wid] for wid in outer_way_ids if wid in ways]
        ring     = BoundaryConverter._chain_way_segments(segments)
        coords   = [nodes[n] for n in ring if n in nodes]

        BoundaryConverter._validate(coords, f"OSM relation {relation_id}")
        return coords

    @staticmethod
    def _chain_way_segments(segments: List[List[int]]) -> List[int]:
        """
        Given a list of node-id lists (way segments), chain them into
        one continuous ring. Reverses individual segments as needed.
        """
        if not segments:
            return []

        ring = list(segments[0])
        remaining = list(segments[1:])

        while remaining:
            tail = ring[-1]
            matched = False
            for i, seg in enumerate(remaining):
                if seg[0] == tail:
                    ring.extend(seg[1:])
                    remaining.pop(i)
                    matched = True
                    break
                elif seg[-1] == tail:
                    ring.extend(reversed(seg[:-1]))
                    remaining.pop(i)
                    matched = True
                    break
            if not matched:
                # Unchainable — just append remaining nodes
                log.warning("OSM relation: unchainable segment, appending directly.")
                ring.extend(remaining.pop(0))

        return ring

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(coords: List[LatLon], source: str) -> None:
        if len(coords) < 3:
            raise ValueError(f"{source}: need ≥ 3 coordinates, got {len(coords)}.")

        for i, (lat, lon) in enumerate(coords):
            if not (-90 <= lat <= 90):
                raise ValueError(f"{source} point {i}: lat={lat} out of [-90,90].")
            if not (-180 <= lon <= 180):
                raise ValueError(f"{source} point {i}: lon={lon} out of [-180,180].")

        # Warn if polygon looks geographically huge (>10 km diagonal)
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        dlat = (max(lats) - min(lats)) * 111_000   # approx metres
        dlon = (max(lons) - min(lons)) * 111_000 * math.cos(math.radians(sum(lats)/len(lats)))
        diag = math.sqrt(dlat**2 + dlon**2)
        if diag > 10_000:
            log.warning(
                "%s: boundary diagonal is %.0f m — equirectangular projection "
                "accuracy degrades beyond ~5 km. Consider dividing the area.", source, diag
            )

        log.info("%s: parsed %d boundary vertices (diagonal ≈ %.0f m).",
                 source, len(coords), diag)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Convert GeoJSON/GPX/KML/OSM boundary → ASV mission JSON"
    )
    parser.add_argument("input",  help="Input file path")
    parser.add_argument("-o", "--output", default="mission_boundary.json",
                        help="Output JSON file (default: mission_boundary.json)")
    parser.add_argument("--mission-name", default="asv_mission",
                        help="Mission identifier string")
    parser.add_argument("--gs-lat", type=float,
                        help="Ground station latitude (optional)")
    parser.add_argument("--gs-lon", type=float,
                        help="Ground station longitude (optional)")
    parser.add_argument("--osm-way", type=int,
                        help="OSM way ID to extract (for .osm files)")
    parser.add_argument("--osm-relation", type=int,
                        help="OSM relation ID to extract (for .osm files)")
    args = parser.parse_args()

    coords = BoundaryConverter.load(
        args.input,
        osm_way=args.osm_way,
        osm_relation=args.osm_relation,
    )

    gs = None
    if args.gs_lat is not None and args.gs_lon is not None:
        gs = (args.gs_lat, args.gs_lon)

    BoundaryConverter.save(
        coords,
        output_path=args.output,
        mission_name=args.mission_name,
        ground_station=gs,
    )
    print(f"✓ Converted {len(coords)} vertices → {args.output}")


if __name__ == "__main__":
    main()
