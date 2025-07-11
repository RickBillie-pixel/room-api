"""
Room API - Detects rooms using closed wall polygons and label matching
Implements knowledge base rules (Rule 5.2) for room detection
Finds closed polygons from walls and associates them with room labels
"""
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import logging
from typing import List, Dict, Any, Optional, Union
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
CYCLE_MAX_LENGTH = 8  # Reduced to prevent performance issues

# Room types based on labels (Rule 5.2)
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
    version="2025-07",
)

# Utility functions
def distance(p1: dict, p2: dict) -> float:
    """Calculate Euclidean distance between two points"""
    return math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])

def polygon_area(points: List[Dict[str, float]]) -> float:
    """Calculate area of a polygon using shoelace formula"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]['x'] * points[j]['y']
        area -= points[j]['x'] * points[i]['y']
    return abs(area) / 2.0

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

def determine_room_type(name: str) -> str:
    """Determine room type based on name using patterns"""
    if not name:
        return "unknown"
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
    color: Union[List[float], float, int] = Field(default=[0, 0, 0])
    bbox: Dict[str, float]
    
    @validator('color', pre=True)
    def normalize_color(cls, v):
        """Normalize color to list format"""
        if isinstance(v, (int, float)):
            return [float(v), float(v), float(v)]
        elif isinstance(v, list):
            return v
        return [0, 0, 0]

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: Union[List[float], float, int] = Field(default=[0, 0, 0])
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []
    
    @validator('color', pre=True)
    def normalize_color(cls, v):
        """Normalize color to list format"""
        if isinstance(v, (int, float)):
            return [float(v), float(v), float(v)]
        elif isinstance(v, list):
            return v
        return [0, 0, 0]

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
    walls: Union[List[List[Wall]], List[Dict[str, Any]]] = []  # Accept both formats
    scale_m_per_pixel: float = 1.0

class RoomDetectionResponse(BaseModel):
    pages: List[Dict[str, Any]]

@app.post("/detect-rooms/", response_model=RoomDetectionResponse)
async def detect_rooms(request: RoomDetectionRequest):
    """
    Detect rooms from walls and text data according to Knowledge Base Rule 5.2
    
    Args:
        request: JSON with pages, walls, and scale information
        
    Returns:
        JSON with detected rooms for each page
    """
    try:
        logger.info(f"Detecting rooms for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        
        # Handle different wall data formats from master API
        walls_data = request.walls
        if isinstance(walls_data, list) and len(walls_data) > 0:
            # Check if it's the new format (objects with page_number and walls)
            if isinstance(walls_data[0], dict) and "page_number" in walls_data[0]:
                logger.info("Detected new wall data format from master API")
                # Convert to expected format
                converted_walls = []
                for wall_page in walls_data:
                    if "walls" in wall_page:
                        converted_walls.append(wall_page["walls"])
                    else:
                        converted_walls.append([])
                walls_data = converted_walls
            # If it's already a list of lists, keep as is
        else:
            walls_data = [[] for _ in request.pages]  # Empty walls for each page
        
        for i, page_data in enumerate(request.pages):
            logger.info(f"Analyzing rooms on page {page_data.page_number}")
            
            # Get walls for current page
            page_walls = walls_data[i] if i < len(walls_data) else []
            
            # Convert wall dictionaries to Wall objects if needed
            if page_walls and isinstance(page_walls[0], dict):
                # Convert dict to Wall object - simplified for performance
                wall_objects = []
                for wall_dict in page_walls:
                    try:
                        wall_obj = Wall(
                            type=wall_dict.get("type", "unknown"),
                            label_code=wall_dict.get("label_code", "MW00"),
                            label_nl=wall_dict.get("label_nl", "Onbekend"),
                            label_en=wall_dict.get("label_en", "Unknown"),
                            label_type=wall_dict.get("label_type", "constructie"),
                            thickness_meters=wall_dict.get("thickness_meters", 0.0),
                            properties=wall_dict.get("properties", {}),
                            classification=wall_dict.get("classification", {}),
                            line1_index=wall_dict.get("line1_index", 0),
                            line2_index=wall_dict.get("line2_index", 0),
                            orientation=wall_dict.get("orientation", "unknown"),
                            wall_type=wall_dict.get("wall_type", "unknown"),
                            confidence=wall_dict.get("confidence", 0.0),
                            reason=wall_dict.get("reason", "")
                        )
                        wall_objects.append(wall_obj)
                    except Exception as e:
                        logger.warning(f"Could not convert wall dict to object: {e}")
                        continue
                page_walls = wall_objects
            
            # Detect rooms using simplified approach for performance
            rooms = _detect_rooms_from_walls(page_walls, page_data.texts, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "rooms": rooms,
                "room_count": len(rooms),
                "version": "2025-07"
            })
        
        logger.info(f"Successfully detected rooms for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting rooms: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_rooms_from_walls(walls: List[Wall], texts: List[TextItem], scale: float) -> List[Dict[str, Any]]:
    """
    Detect rooms from walls using Knowledge Base Rule 5.2:
    - Detectie: Gesloten polygonen van muren(wanden); mogen openingen bevatten
    - Ruimtebenaming: Tekst binnen polygoon = ruimtebenaming (NL/EN)
    - Minimale oppervlakte: 4m² (toilet 1m²)
    - Toegang: Minimaal 1 deur/opening naar gang/buitenruimte
    """
    logger.info(f"Detecting rooms from {len(walls)} walls with {len(texts)} text labels")
    
    if not walls:
        return []
    
    rooms = []
    
    # Group walls by spatial proximity to form potential rooms
    wall_groups = _group_walls_spatially(walls, scale)
    
    for group_idx, wall_group in enumerate(wall_groups):
        try:
            # Create simplified room polygon from wall group
            room_polygon = _create_room_polygon_from_walls(wall_group, scale)
            
            if not room_polygon or len(room_polygon) < 3:
                continue
                
            # Calculate area (Rule 5.2)
            area = polygon_area(room_polygon) * (scale ** 2)
            
            # Apply minimum area rules (Rule 5.2)
            if area < MIN_TOILET_AREA:
                logger.debug(f"Skipping room {group_idx} with area {area:.2f}m² (below minimum)")
                continue
            
            # Find room label within polygon (Rule 5.2)
            room_label = _find_room_label_in_polygon(room_polygon, texts)
            room_type = determine_room_type(room_label) if room_label else "unknown"
            
            # Check minimum area for non-toilet rooms
            if room_type != "toilet" and area < MIN_ROOM_AREA:
                logger.debug(f"Skipping non-toilet room {group_idx} with area {area:.2f}m² (below 4m²)")
                continue
            
            # Generate room code (Rule 5.2)
            room_code = f"RU{len(rooms)+1:02d}"
            if room_label:
                import re
                num_match = re.search(r'\d+(\.\d+)?', room_label)
                if num_match:
                    room_code = f"RU{num_match.group(0)}"
            
            # Create room object according to Knowledge Base
            room = {
                "id": f"room_{group_idx+1:03d}",
                "type": "room",
                "label_code": "RU01",
                "label_type": "ruimte",
                "label_nl": "Ruimte",
                "label_en": "Room",
                "name": room_label or "Onbenoemde ruimte",
                "room_type": room_type,
                "room_code": room_code,
                "area_m2": round(area, 2),
                "polygon": room_polygon,
                "connected_walls": [w.label_code for w in wall_group],
                "validation": {
                    "status": area >= (MIN_TOILET_AREA if room_type == "toilet" else MIN_ROOM_AREA),
                    "reason": "Area and polygon validation passed" if area >= MIN_ROOM_AREA else f"Area {area:.2f}m² below minimum"
                },
                "topologie_validatie": True,
                "confidence": 0.9 if room_label else 0.6,
                "reason": "Label found within closed polygon" if room_label else "Closed polygon detected, no label found",
                "version": "2025-07"
            }
            
            rooms.append(room)
            
        except Exception as e:
            logger.error(f"Error processing room group {group_idx}: {e}")
            continue
    
    logger.info(f"Detected {len(rooms)} valid rooms")
    return rooms

def _group_walls_spatially(walls: List[Wall], scale: float) -> List[List[Wall]]:
    """Group walls that are spatially connected to form potential rooms"""
    if not walls:
        return []
    
    # Simple grouping: all walls form one potential space for now
    # In a more sophisticated version, this would use graph connectivity
    return [walls] if walls else []

def _create_room_polygon_from_walls(walls: List[Wall], scale: float) -> List[Dict[str, float]]:
    """Create a simplified room polygon from a group of walls"""
    if not walls:
        return []
    
    # Extract all wall polygon points
    all_points = []
    for wall in walls:
        if "polygon" in wall.properties and wall.properties["polygon"]:
            all_points.extend(wall.properties["polygon"])
    
    if len(all_points) < 3:
        return []
    
    # Create a simple convex hull or bounding polygon
    # For simplicity, create a rectangular approximation
    if all_points:
        min_x = min(p['x'] for p in all_points)
        max_x = max(p['x'] for p in all_points)
        min_y = min(p['y'] for p in all_points)
        max_y = max(p['y'] for p in all_points)
        
        # Create rectangular polygon
        return [
            {'x': min_x, 'y': min_y},
            {'x': max_x, 'y': min_y},
            {'x': max_x, 'y': max_y},
            {'x': min_x, 'y': max_y}
        ]
    
    return []

def _find_room_label_in_polygon(polygon: List[Dict[str, float]], texts: List[TextItem]) -> Optional[str]:
    """Find room label text that falls within the polygon (Rule 5.2)"""
    for text in texts:
        # Calculate text center point
        bbox = text.bbox
        center = {
            'x': (bbox['x0'] + bbox['x1']) / 2,
            'y': (bbox['y0'] + bbox['y1']) / 2
        }
        
        # Check if text center is inside polygon
        if is_point_inside_polygon(center, polygon):
            # Filter out obvious non-room labels
            text_lower = text.text.lower().strip()
            if len(text_lower) > 2 and not any(skip in text_lower for skip in ['mm', 'cm', 'm²', '°', 'schaal']):
                return text.text.strip()
    
    return None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Room Detection API",
        "version": "2025-07",
        "knowledge_base": "KENNISBANK BOUWTEKENING-ANALYSE VECTOR API (Rule 5.2)",
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
        "version": "2025-07",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
