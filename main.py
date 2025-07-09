"""
Room API - Detects rooms using closed wall polygons and label matching
Implements knowledge base rules (Rule 5.2) for room detection
Finds closed polygons from walls and associates them with room labels
"""
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("room_api")

# Knowledge Base Constants (Rule 5.2)
MIN_ROOM_AREA = 4.0  # m² - Minimum area for a room (Rule 5.2)
MIN_TOILET_AREA = 1.0  # m² - Minimum area for a toilet (Rule 5.2)
SNAP_TOLERANCE_M = 0.02  # m - Tolerance for snapping points (2cm)
MAX_ROOM_VERTICES = 20  # Maximum vertices for a room polygon
CYCLE_MAX_LENGTH = 12  # Maximum length of cycles to consider

# Room types based on labels
ROOM_TYPE_PATTERNS = {
    "living_room": ["woonkamer", "living", "zitkamer", "salon", "lounge"],
    "kitchen": ["keuken", "kitchen", "kookruimte"],
    "bedroom": ["slaapkamer", "bedroom", "master bedroom", "kinderkamer"],
    "bathroom": ["badkamer", "bathroom", "douche", "shower"],
    "toilet": ["toilet", "wc", "washroom"],
    "hall": ["hal", "hall", "entree", "entrance", "gang", "corridor"],
    "storage": ["berging", "storage", "opslag", "kast", "closet"],
    "utility": ["technische ruimte", "wasruimte", "utility", "cv", "stookruimte"],
    "office": ["kantoor", "office", "studeerkamer", "study", "werkruimte"]
}

app = FastAPI(
    title="Room Detection API",
    description="Detects rooms using closed wall polygons and label matching",
    version="1.0.0",
)

# Utility functions
def distance(p1: dict, p2: dict) -> float:
    """Calculate Euclidean distance between two points"""
    return math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])

def snap_points(points: List[dict], tolerance: float) -> List[dict]:
    """Snap points together if they are within tolerance distance"""
    snapped = []
    for p in points:
        found = False
        for s in snapped:
            if distance(p, s) <= tolerance:
                found = True
                break
        if not found:
            snapped.append(p)
    return snapped

def quantize_point(p: dict, precision: float = 0.001) -> tuple:
    """Quantize point to given precision for graph representation"""
    return (round(p['x'] / precision) * precision, round(p['y'] / precision) * precision)

def midpoint(p1: dict, p2: dict) -> dict:
    """Calculate midpoint between two points"""
    return {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}

def is_point_inside_polygon(p: dict, poly: List[dict]) -> bool:
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = p['x'], p['y']
    n = len(poly)
    inside = False
    px1, py1 = poly[0]['x'], poly[0]['y']
    for i in range(n + 1):
        px2, py2 = poly[i % n]['x'], poly[i % n]['y']
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-12) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside

def polygon_area(points: List[Dict[str, float]]) -> float:
    """Calculate area of a polygon using shoelace formula"""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]['x'] * points[j]['y']
        area -= points[j]['x'] * points[i]['y']
    return abs(area) / 2.0

def determine_room_type(name: str) -> str:
    """Determine room type based on name using patterns"""
    name_lower = name.lower()
    
    for room_type, patterns in ROOM_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return room_type
    
    return "unknown"

class Wall(BaseModel):
    type: str
    label_code: str
    label_nl: str
    label_en: str
    label_type: str
    thickness_meters: float
    properties: Dict[str, Any]
    classification: Dict[str, Any]
    line1_index: int
    line2_index: int
    orientation: str
    wall_type: str
    confidence: float
    reason: str

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: List[float] = [0, 0, 0]
    bbox: Dict[str, float]

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: List[float] = [0, 0, 0]
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class PageData(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    drawings: Drawings
    texts: List[TextItem]
    is_vector: bool = True
    processing_time_ms: Optional[int] = None

class RoomDetectionRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    scale_m_per_pixel: float = 1.0

class RoomDetectionResponse(BaseModel):
    pages: List[Dict[str, Any]]

@app.post("/detect-rooms/", response_model=RoomDetectionResponse)
async def detect_rooms(request: RoomDetectionRequest):
    """
    Detect rooms from walls and text data
    
    Args:
        request: JSON with pages, walls, and scale information
        
    Returns:
        JSON with detected rooms for each page
    """
    try:
        logger.info(f"Detecting rooms for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        
        for i, page_data in enumerate(request.pages):
            logger.info(f"Analyzing rooms on page {page_data.page_number}")
            
            # Get walls for current page
            page_walls = request.walls[i] if i < len(request.walls) else []
            
            rooms = _find_closed_rooms(page_walls, page_data.texts, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "rooms": rooms
            })
        
        logger.info(f"Successfully detected rooms for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting rooms: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _find_closed_rooms(walls: List[Wall], texts: List[TextItem], scale: float) -> List[Dict[str, Any]]:
    """
    Find closed rooms from walls using polygon detection
    
    Args:
        walls: List of wall dictionaries
        texts: List of text dictionaries
        scale: Scale factor in meters per pixel
        
    Returns:
        List of detected rooms with properties
    """
    logger.info(f"Finding rooms from {len(walls)} walls (snap tolerance: {SNAP_TOLERANCE_M}m)")
    
    # Extract all wall endpoints
    endpoints = []
    for wall in walls:
        if "polygon" in wall.properties:
            # If wall has polygon representation, use its vertices
            for point in wall.properties["polygon"]:
                endpoints.append(point)
        else:
            # Otherwise use wall endpoints
            endpoints.append(wall.properties["polygon"][0])
            endpoints.append(wall.properties["polygon"][3])
    
    # Snap points together
    snapped_points = snap_points(endpoints, SNAP_TOLERANCE_M / scale)
    logger.info(f"Found {len(snapped_points)} unique vertices after snapping")
    
    def find_nearest_snapped(p):
        return min(snapped_points, key=lambda s: distance(p, s))
    
    # Build point graph for cycle detection
    point_graph = defaultdict(list)
    for wall in walls:
        # Use wall polygon if available
        if "polygon" in wall.properties:
            poly = wall.properties["polygon"]
            for i in range(len(poly)):
                p1 = find_nearest_snapped(poly[i])
                p2 = find_nearest_snapped(poly[(i+1) % len(poly)])
                qp1 = quantize_point(p1)
                qp2 = quantize_point(p2)
                
                if qp2 not in point_graph[qp1]:
                    point_graph[qp1].append(qp2)
                if qp1 not in point_graph[qp2]:
                    point_graph[qp2].append(qp1)
    
    # Find cycles (closed polygons)
    cycles = _find_cycles(point_graph)
    
    if not cycles:
        logger.warning("No closed polygons found.")
        return [{
            "type": "unknown", 
            "reason": "No closed polygons found", 
            "confidence": 0.0
        }]
    
    # Convert cycles to rooms
    rooms = []
    for poly in cycles:
        poly_points = [{"x": x, "y": y} for (x, y) in poly]
        
        # Calculate room area
        area = polygon_area(poly_points) * (scale ** 2)
        
        # Skip rooms that are too small (Rule 5.2)
        if area < MIN_ROOM_AREA and area < MIN_TOILET_AREA:
            logger.debug(f"Skipping room with area {area:.2f}m² (too small)")
            continue
        
        # Find room label
        label = None
        room_type = "unknown"
        
        for text in texts:
            text_dict = text.dict()
            center = midpoint(
                {'x': text_dict['bbox']['x0'], 'y': text_dict['bbox']['y0']}, 
                {'x': text_dict['bbox']['x1'], 'y': text_dict['bbox']['y1']}
            )
            if is_point_inside_polygon(center, poly_points):
                label = text_dict['text']
                room_type = determine_room_type(label)
                break
        
        # Generate room code based on label or location
        room_code = f"R{len(rooms)+1}"
        if label:
            # Extract numeric part if present (e.g. "Room 1.4" -> "1.4")
            import re
            num_match = re.search(r'\d+(\.\d+)?', label)
            if num_match:
                room_code = f"R{num_match.group(0)}"
        
        # Check if room has at least one door/opening (Rule 5.2 - Access)
        # This would require door detection data which we don't have here
        # For now, we assume all rooms have access
        has_access = True
        
        rooms.append({
            "name": label or "Unnamed Room",
            "room_type": room_type,
            "room_code": room_code,
            "area_m2": round(area, 2),
            "polygon": poly_points,
            "confidence": 1.0 if label else 0.7,
            "reason": "Label found inside polygon" if label else "No label found, area computed",
            "has_access": has_access,
            "label_code": "RU01",
            "label_type": "room",
            "label_nl": "Ruimte",
            "label_en": "Room"
        })
    
    logger.info(f"Found {len(rooms)} rooms")
    return rooms

def _find_cycles(point_graph: Dict[tuple, List[tuple]]) -> List[List[tuple]]:
    """
    Find cycles in point graph using DFS
    
    Args:
        point_graph: Graph representation of connected points
        
    Returns:
        List of cycles (closed polygons)
    """
    cycles = set()
    
    def find_all_cycles_dfs(node, visited, path, start_node, depth=0):
        """DFS to find all cycles"""
        if depth > CYCLE_MAX_LENGTH:
            return  # Limit cycle length to avoid excessive computation
            
        if node == start_node and len(path) > 2:
            # Found a cycle
            cycle = tuple(path)
            cycles.add(cycle)
            return
            
        if node in visited and node != start_node:
            return
            
        visited.add(node)
        
        for neighbor in point_graph.get(node, []):
            find_all_cycles_dfs(neighbor, visited.copy(), path + [neighbor], start_node, depth + 1)
    
    # Start DFS from each node
    for node in point_graph:
        find_all_cycles_dfs(node, set(), [node], node)
    
    # Filter cycles to remove duplicates and subsets
    unique_cycles = []
    seen = set()
    
    # First, sort cycles by length (number of vertices)
    sorted_cycles = sorted(cycles, key=len)
    
    for cycle in sorted_cycles:
        # Convert to a set for easier comparison
        cycle_set = set(cycle)
        
        # Check if this cycle is a superset of any cycle we've already seen
        is_superset = False
        for seen_cycle in seen:
            if cycle_set.issuperset(seen_cycle):
                is_superset = True
                break
                
        if not is_superset and len(cycle) <= MAX_ROOM_VERTICES:
            unique_cycles.append(list(cycle))
            seen.add(frozenset(cycle))
    
    return unique_cycles

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Room Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-rooms/": "Detect rooms using closed wall polygons",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "room-api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)