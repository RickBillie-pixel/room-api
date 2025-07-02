"""
Room API - Detects rooms using closed wall polygons and label matching
Finds closed polygons from walls and associates them with room labels
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
import os
from typing import List, Dict, Any
from collections import defaultdict

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import (
    distance, snap_points, quantize_point, midpoint, 
    is_point_inside_polygon, polygon_area, SNAP_TOLERANCE_M
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("room_api")

app = FastAPI(
    title="Room Detection API",
    description="Detects rooms using closed wall polygons and label matching",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class Wall(BaseModel):
    p1: Dict[str, float]
    p2: Dict[str, float]
    wall_thickness: float
    wall_length: float
    wall_type: str
    confidence: float
    reason: str

class RoomDetectionRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    scale_m_per_pixel: float = 1.0

@app.post("/detect-rooms/")
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
            
            # Convert walls to list of dictionaries
            walls_dict = [wall.dict() for wall in request.walls[i]]
            
            rooms = _find_closed_rooms(walls_dict, page_data.texts, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "rooms": rooms
            })
        
        logger.info(f"Successfully detected rooms for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting rooms: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _find_closed_rooms(walls: List[Dict[str, Any]], texts: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
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
        endpoints.append(wall['p1'])
        endpoints.append(wall['p2'])
    
    # Snap points together
    snapped_points = snap_points(endpoints, SNAP_TOLERANCE_M / scale)
    
    def find_nearest_snapped(p):
        return min(snapped_points, key=lambda s: distance(p, s))
    
    # Build point graph for cycle detection
    point_graph = defaultdict(list)
    for wall in walls:
        sp1 = find_nearest_snapped(wall['p1'])
        sp2 = find_nearest_snapped(wall['p2'])
        qp1 = quantize_point(sp1)
        qp2 = quantize_point(sp2)
        
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
        
        # Find room label
        label = None
        for t in texts:
            center = midpoint(
                {'x': t['bbox']['x0'], 'y': t['bbox']['y0']}, 
                {'x': t['bbox']['x1'], 'y': t['bbox']['y1']}
            )
            if is_point_inside_polygon(center, poly_points):
                label = t['text']
                break
        
        # Calculate room area
        area = polygon_area(poly_points) * (scale ** 2)
        
        rooms.append({
            "name": label or "Unknown",
            "area_m2": round(area, 2),
            "polygon": poly_points,
            "confidence": 1.0 if label else 0.7,
            "reason": "Label found inside polygon" if label else "No label found, area computed"
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
    
    for start in point_graph:
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            for neighbor in point_graph[current]:
                if neighbor == path[0] and len(path) > 2:
                    cycle = tuple(path)
                    if len(set(path)) == len(path):  # No repeated vertices
                        cycles.add(cycle)
                elif neighbor not in path:
                    if len(path) < 12:  # Limit cycle length
                        stack.append((neighbor, path + [neighbor]))
    
    # Remove duplicate cycles
    unique_cycles = []
    seen = set()
    for cyc in cycles:
        key = tuple(sorted(cyc))
        if key not in seen:
            seen.add(key)
            unique_cycles.append(list(cyc))
    
    return unique_cycles

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "room-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 