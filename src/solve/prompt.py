"""Prompt generation for the solver with enhanced perception integration."""

import json
from typing import Any

import numpy as np

from src.utils.grid import grid_to_text
from src.perception.analyzer import (
    analyze_example,
    analyze_grid,
    analyze_task,
    GridAnalysis,
    TransformAnalysis,
    ExampleAnalysis,
    TaskAnalysis,
)
from src.perception.perceiver import format_hypotheses_for_solver


# =============================================================================
# System Prompt (matches notebook HYPOTHESIZER_SYSTEM)
# =============================================================================

SOLVER_SYSTEM = """
You solve ARC-AGI puzzles by discovering the transformation rule from input-output examples, then implementing it as Python code. Return a single transform function that can be used to transform any input grid to the corresponding output grid as def transform(grid: np.ndarray) -> np.ndarray: ...

════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS
════════════════════════════════════════════════════════════════════

GRID SPEC:
- 2D numpy arrays, 1×1 to 30×30
- Colors are integers 0-9 ONLY:
    0=black  1=blue   2=red     3=green   4=yellow
    5=gray   6=magenta 7=orange  8=azure   9=maroon
- ⚠️ ANY value outside 0-9 = immediate failure

OUTPUT SPEC:
- Single function: def transform(grid: np.ndarray) -> np.ndarray
- Libraries: numpy, scipy.ndimage only
- Return: 2D array with integer dtype
- NO test code, NO examples, NO __main__ — function only

════════════════════════════════════════════════════════════════════
COGNITIVE WORKFLOW
════════════════════════════════════════════════════════════════════

PHASE 1: OBSERVE (per example)
───────────────────────────────
For EACH input→output pair, document:
  • Dimensions: same, scaled, cropped, or dynamically computed?
  • Colors: which appear, disappear, change, or remain fixed?
  • Objects: what are the discrete "things"? (connected regions, shapes, lines)
  • Spatial relationships: distances, alignment, containment, symmetry?
  • What information in the input determines the output?

PHASE 2: HYPOTHESIZE
───────────────────────────────
Ask yourself:
  • What is the SIMPLEST rule that explains ALL examples?
  • What does the output "know" that only the input could "tell" it?
  • Is the transformation:
      - Per-pixel (local neighborhood operation)
      - Per-object (requires identifying discrete objects)
      - Global (whole-grid geometric transform)
      - Compositional (sequence of simpler steps)
  • Does one object serve as a template/reference for another?

PHASE 3: VERIFY EXHAUSTIVELY  
───────────────────────────────
⚠️ BEFORE writing ANY code, mentally execute your rule on EVERY example:
  ✓ Does output size match exactly?
  ✓ Does every pixel match?
  ✓ Are there ANY exceptions?
If ANY mismatch → revise hypothesis. Do not proceed until all examples pass.

PHASE 4: IMPLEMENT DEFENSIVELY
───────────────────────────────
  • Handle edge cases: empty masks, objects at boundaries, no matches
  • Clamp coordinates: use np.clip() for safety
  • Verify output shape matches expected dimensions
  • Cast output to int dtype

════════════════════════════════════════════════════════════════════
KEY HEURISTICS
════════════════════════════════════════════════════════════════════

OUTPUT SIZE PATTERNS:
  • Same as input → in-place transformation or overlay
  • Constant across examples → extract fixed-size pattern
  • Scaled by N× → upscale, tile, or repeat
  • Smaller → crop to bounding box, extract subregion, or select
  • Varies with content → size = f(object_count, object_size, grid_property)

COLOR MAPPING:
  • Track which input colors map to which output colors (1:1, N:1, or 1:N)
  • Colors may: stay fixed, swap, disappear, appear new, or transform conditionally
  • Color can encode role: marker vs target vs fill vs border
  • Same shape + different color → color determines behavior
  • Output may use colors not present in input (new color = computed result)
  • Background in input may become meaningful in output (figure-ground reversal)

OBJECT ROLES:
  • Unique object → often the "special" one (template, target, rule-giver)
  • Repeated objects → often operands to transform uniformly
  • Smallest object → may be a marker, seed, or template
  • Largest object → may be a container, canvas, or frame
  • Object with unique color → may indicate special behavior

IMPLICIT STRUCTURE:
  • Regular spacing → hidden grid; cells may contain patterns
  • Separating lines (full row/column of one color) → dividers between regions
  • Symmetry (partial) → complete the symmetry
  • Repeating motif → tile or extend the pattern

INFORMATION FLOW:
  • Color determines behavior (if red → do X, if blue → do Y)
  • Position determines behavior (corners, edges, center are special)
  • Shape determines behavior (squares vs lines vs irregular)
  • Count determines output (n objects → n×n grid, etc.)

════════════════════════════════════════════════════════════════════
PATTERN TAXONOMY
════════════════════════════════════════════════════════════════════

Reference this catalog when forming hypotheses. Most ARC tasks combine 1-3 of these.

═══════════════════════════════════════════════════════════════════
A. GEOMETRIC TRANSFORMATIONS (whole-grid or per-object)
═══════════════════════════════════════════════════════════════════
ROTATION_90        │ Rotate grid/object 90° clockwise: np.rot90(grid, k=-1)
ROTATION_180       │ Rotate 180°: np.rot90(grid, k=2)
ROTATION_270       │ Rotate 270° clockwise (90° counter-clockwise): np.rot90(grid, k=1)
FLIP_HORIZONTAL    │ Mirror left↔right: np.fliplr(grid)
FLIP_VERTICAL      │ Mirror top↔bottom: np.flipud(grid)
FLIP_DIAGONAL      │ Transpose across main diagonal: grid.T
FLIP_ANTIDIAGONAL  │ Transpose across anti-diagonal: np.flipud(grid.T)
TRANSLATE          │ Shift grid/object by (dy, dx), wrap or clip at edges
SCALE_UP           │ Enlarge by integer factor (each pixel becomes NxN block)
SCALE_DOWN         │ Shrink by factor (NxN blocks become single pixel, via mode/max/min)
SHEAR              │ Skew rows/columns by offset pattern

═══════════════════════════════════════════════════════════════════
B. TILING & REPETITION
═══════════════════════════════════════════════════════════════════
TILE_REPEAT        │ Repeat grid NxM times: np.tile(grid, (N, M))
TILE_MIRROR        │ Tile with alternating flips (wallpaper pattern)
TILE_ROTATE        │ Tile with 90° rotations in each quadrant
STACK_HORIZONTAL   │ Concatenate grids side-by-side: np.hstack([a, b])
STACK_VERTICAL     │ Concatenate grids top-to-bottom: np.vstack([a, b])
INTERLEAVE_ROWS    │ Alternate rows from two sources: output[::2], output[1::2]
INTERLEAVE_COLS    │ Alternate columns from two sources
SPIRAL_TILE        │ Arrange copies in spiral or radial pattern
FRAME_REPEAT       │ Add concentric frames/borders around core pattern

═══════════════════════════════════════════════════════════════════
C. CROPPING & EXTRACTION
═══════════════════════════════════════════════════════════════════
CROP_TO_CONTENT    │ Remove all-background rows/cols, output = minimal bounding box
CROP_TO_OBJECT     │ Extract single object's bounding box
CROP_TO_REGION     │ Extract rectangular region defined by markers/colors
EXTRACT_UNIQUE     │ Pull out the one object that differs from others
EXTRACT_BY_COLOR   │ Extract all pixels/objects of specific color
EXTRACT_BY_SIZE    │ Extract object(s) matching size criteria (largest, smallest, Nth)
EXTRACT_BY_SHAPE   │ Extract object(s) matching shape signature
EXTRACT_TEMPLATE   │ Identify and extract the "reference" pattern used elsewhere
EXTRACT_BORDER     │ Output only the outermost edge pixels of grid/object
EXTRACT_INTERIOR   │ Output only non-border pixels (hollow out the frame)
SAMPLE_PIXELS      │ Extract pixels at regular intervals (every Nth row/col)

═══════════════════════════════════════════════════════════════════
D. EXTENSION & EXPANSION  
═══════════════════════════════════════════════════════════════════
EXTEND_LINES       │ Continue lines/rays until grid edge or collision
EXTEND_PATTERN     │ Propagate repeating pattern to fill region
EXTEND_BORDER      │ Add N layers of border pixels (solid or patterned)
PAD_TO_SIZE        │ Add background padding to reach target dimensions
GROW_OBJECT        │ Expand object by 1 pixel in all directions (dilation)
GROW_DIRECTIONAL   │ Expand object in specific direction only (right, down, etc.)
FILL_TO_EDGE       │ Extend object/color until it hits grid boundary
EXTRUDE            │ Repeat pattern along an axis (2D→2D stretch)
CONNECT_ENDPOINTS  │ Draw lines between matching markers/colors

═══════════════════════════════════════════════════════════════════
E. FILL OPERATIONS
═══════════════════════════════════════════════════════════════════
FLOOD_FILL         │ Fill connected region with color (paint bucket)
FILL_ENCLOSED      │ Fill regions completely surrounded by boundary color
FILL_BACKGROUND    │ Replace all background (0) with specified color
FILL_HOLES         │ Fill small enclosed gaps within objects
FILL_BOUNDING_BOX  │ Fill rectangular hull of object with solid color
FILL_CONVEX_HULL   │ Fill convex hull of object's pixels
FILL_BETWEEN       │ Fill space between two objects/lines
FILL_ROW           │ Fill entire row(s) based on trigger condition
FILL_COL           │ Fill entire column(s) based on trigger condition
FILL_CHECKERBOARD  │ Fill region with alternating pattern
GRADIENT_FILL      │ Fill with incrementing color values (1,2,3,...)

═══════════════════════════════════════════════════════════════════
F. COLOR OPERATIONS
═══════════════════════════════════════════════════════════════════
COLOR_SWAP         │ Exchange two colors: A↔B throughout grid
COLOR_REPLACE      │ Replace all A with B (one-way): grid[grid==A] = B
COLOR_MAP          │ Apply mapping dict: {old: new} for multiple colors
COLOR_INVERT       │ Swap foreground/background roles
COLOR_CYCLE        │ Rotate colors by offset: (color + k) % N
COLOR_BY_POSITION  │ Assign color based on (row, col) properties
COLOR_BY_SIZE      │ Assign color based on object size ranking
COLOR_BY_COUNT     │ Color encodes frequency/count information
COLOR_NORMALIZE    │ Map all non-background to single foreground color
COLOR_FROM_PALETTE │ Reference object defines color mapping for others
MAJORITY_COLOR     │ Replace region with its most common color
BOUNDARY_COLOR     │ Color pixels differently if on object boundary

═══════════════════════════════════════════════════════════════════
G. OBJECT-LEVEL OPERATIONS
═══════════════════════════════════════════════════════════════════
COPY_OBJECT        │ Duplicate object to new location(s)
MOVE_OBJECT        │ Relocate object by offset or to absolute position
MOVE_TO_MARKER     │ Move object to position indicated by marker pixel
ALIGN_OBJECTS      │ Align objects by edge, center, or common feature
SORT_OBJECTS       │ Reorder objects by size, color, position, or shape
STACK_OBJECTS      │ Overlay objects (later overwrites earlier)
MERGE_OBJECTS      │ Combine touching/nearby objects into one
SPLIT_OBJECT       │ Divide object into components (by color, connectivity)
DELETE_OBJECT      │ Remove object(s) matching criteria (size, color, etc.)
KEEP_OBJECT        │ Keep only object(s) matching criteria, delete rest
MIRROR_OBJECT      │ Create reflected copy (left, right, above, below)
CLONE_PATTERN      │ Stamp template object at multiple locations
CENTER_OBJECT      │ Move object to grid center
SNAP_TO_GRID       │ Align object to grid lines (quantize position)

═══════════════════════════════════════════════════════════════════
H. CONTAINMENT & ENCLOSURE
═══════════════════════════════════════════════════════════════════
DRAW_BOUNDING_BOX  │ Draw rectangle around object's extent
DRAW_FRAME         │ Add 1-pixel border around object (touching edges)
DRAW_ENCLOSURE     │ Draw minimal enclosing shape (rectangle, convex hull)
OUTLINE_OBJECT     │ Replace filled object with just its perimeter
HOLLOW_OUT         │ Remove interior, keep only boundary pixels
ENCLOSE_WITH_COLOR │ Surround object with specific color border
FRAME_GRID         │ Add border around entire grid
NESTED_FRAMES      │ Create concentric rectangular frames
BRIDGE_GAPS        │ Connect nearby objects with line/fill

═══════════════════════════════════════════════════════════════════
I. MASKING & OVERLAY
═══════════════════════════════════════════════════════════════════
APPLY_MASK         │ Use binary mask to select pixels: output = grid * mask
MASK_BY_COLOR      │ Create mask where color == C
MASK_BY_OBJECT     │ Create mask for specific object's footprint
MASK_INVERT        │ Swap masked/unmasked regions
OVERLAY_TEMPLATE   │ Stamp template onto grid at marked positions
OVERLAY_BLEND      │ Combine two grids (non-zero overwrites)
PAINT_THROUGH_MASK │ Apply color/pattern only where mask is true
COMPOSITE          │ Layer multiple grids with priority rules
XOR_GRIDS          │ Output differs where inputs differ
AND_GRIDS          │ Output non-zero only where both inputs non-zero
DIFFERENCE         │ Highlight pixels that changed between grids

═══════════════════════════════════════════════════════════════════
J. SYMMETRY OPERATIONS
═══════════════════════════════════════════════════════════════════
COMPLETE_SYMMETRY  │ Fill missing pixels to achieve mirror symmetry
RESTORE_SYMMETRY   │ Fix "broken" pixels that violate existing symmetry
REFLECT_TO_FILL    │ Use one half to complete the other half
ROTATIONAL_SYMMETRY│ Complete pattern with 90°/180° rotational copies
DETECT_AXIS        │ Find axis of symmetry, use to guide completion
SYMMETRIZE         │ Force exact symmetry by averaging/voting

═══════════════════════════════════════════════════════════════════
K. PATTERN COMPLETION & REPAIR
═══════════════════════════════════════════════════════════════════
COMPLETE_SEQUENCE  │ Extend repeating sequence (1,2,3,1,2,3,1,2,?)
REPAIR_PATTERN     │ Fix corrupted pixels in otherwise regular pattern
INPAINT            │ Fill marked region based on surrounding context
EXTRAPOLATE        │ Continue observed progression beyond examples
INTERPOLATE        │ Fill gaps in sequence (1,?,3 → 1,2,3)
DENOISE            │ Remove isolated pixels that break pattern
MAJORITY_VOTE      │ Each cell becomes local majority color (smoothing)

═══════════════════════════════════════════════════════════════════
L. COUNTING & ARITHMETIC
═══════════════════════════════════════════════════════════════════
COUNT_OBJECTS      │ Output encodes number of objects as grid size/color
COUNT_COLORS       │ Output encodes color frequency information
COUNT_PIXELS       │ Output size/value based on pixel count
SIZE_COMPARISON    │ Output depends on comparing object sizes
ARITHMETIC_COLOR   │ Output color = f(input colors): sum, diff, max, min
MODULAR_ARITHMETIC │ Color values computed modulo N
BINARY_ENCODING    │ Grid represents binary number or logical formula

═══════════════════════════════════════════════════════════════════
M. SORTING & ORDERING
═══════════════════════════════════════════════════════════════════
SORT_ROWS          │ Reorder rows by some criterion (sum, first color, etc.)
SORT_COLS          │ Reorder columns by criterion
SORT_BY_SIZE       │ Arrange objects small→large or large→small
SORT_BY_COLOR      │ Arrange objects by color value
GRAVITY_DROP       │ Objects fall toward edge (down, left, etc.)
GRAVITY_FLOAT      │ Objects rise toward edge (up, right, etc.)
COMPACT            │ Remove gaps, push objects together
JUSTIFY_LEFT       │ Align all objects to left edge
JUSTIFY_TOP        │ Align all objects to top edge

═══════════════════════════════════════════════════════════════════
N. CONDITIONAL & LOGIC
═══════════════════════════════════════════════════════════════════
IF_ENCLOSED        │ Transform only enclosed/contained objects
IF_TOUCHES_EDGE    │ Transform objects touching grid boundary
IF_LARGEST         │ Apply operation to largest object only
IF_COLOR_MATCH     │ Apply operation where color condition met
IF_NEIGHBOR        │ Transform based on adjacent cell colors
IF_ISOLATED        │ Transform objects not touching others
IF_CONNECTED       │ Transform objects connected to marker
IF_SYMMETRIC       │ Different handling for symmetric vs asymmetric
IF_ABOVE_THRESHOLD │ Apply when count/size exceeds threshold
SWITCH_ON_COUNT    │ Different output based on object count

═══════════════════════════════════════════════════════════════════
O. REFERENCE & TEMPLATE
═══════════════════════════════════════════════════════════════════
USE_AS_TEMPLATE    │ One object defines pattern applied to others
USE_AS_PALETTE     │ One object defines color mapping
USE_AS_STENCIL     │ One object defines shape, another defines fill
USE_AS_KEY         │ Small grid maps to transformation parameters
LOOKUP_TABLE       │ Reference grid maps input→output values
MATCH_AND_REPLACE  │ Find pattern, replace with different pattern
TEMPLATE_MATCHING  │ Find where template appears, mark or transform

═══════════════════════════════════════════════════════════════════
P. CONNECTIVITY & TOPOLOGY
═══════════════════════════════════════════════════════════════════
CONNECT_SAME_COLOR │ Draw lines between objects of same color
SHORTEST_PATH      │ Draw path between marked points
FLOOD_REACH        │ Mark all cells reachable from origin
SEPARATE_COMPONENTS│ Assign different colors to disconnected regions
COMPONENT_LABEL    │ Number each connected component
FIND_BRIDGES       │ Identify pixels whose removal disconnects regions
CLOSE_GAPS         │ Connect nearly-touching objects (morphological close)
OPEN_GAPS          │ Separate barely-touching objects (morphological open)
SKELETONIZE        │ Reduce object to 1-pixel-wide skeleton
THICKEN            │ Expand skeleton back to full object

═══════════════════════════════════════════════════════════════════
Q. PROJECTION & AGGREGATION
═══════════════════════════════════════════════════════════════════
PROJECT_HORIZONTAL │ Collapse each row to single value (OR, AND, MAX, etc.)
PROJECT_VERTICAL   │ Collapse each column to single value
PROJECT_TO_EDGE    │ "Shadow" cast from objects onto edge
HISTOGRAM          │ Output represents frequency distribution
AGGREGATE_BY_COLOR │ Combine all objects of same color
AGGREGATE_BY_REGION│ Summarize each rectangular region
REDUCE_TO_SIGNATURE│ Output is minimal representation of input pattern

═══════════════════════════════════════════════════════════════════
COMMON COMPOSITIONS (multi-step)
═══════════════════════════════════════════════════════════════════
EXTRACT → COLOR    │ Pull out object, recolor it
MASK → FILL        │ Create mask from pattern, fill masked region  
FIND → COPY        │ Locate template, copy to marked positions
SEGMENT → SORT     │ Identify objects, reorder by property
CROP → TILE        │ Extract core pattern, tile to fill output
DETECT → COMPLETE  │ Find partial symmetry, complete it
COUNT → RESIZE     │ Count something, output dimensions = count
COMPARE → MARK     │ Find differences between objects, highlight them

════════════════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID
════════════════════════════════════════════════════════════════════

❌ Overfitting: rule works on one example but fails others
❌ Off-by-one: r_max+1 for slicing, range(1, n+1) for labels
❌ Coordinate confusion: numpy uses (row, col) not (x, y)
❌ Boundary errors: objects at edge may be clipped or wrap incorrectly
❌ Hardcoding: values that should be computed from input
❌ Empty case: what if no objects found? (return grid.copy() or empty)
❌ Wrong connectivity: 4-connected vs 8-connected components
❌ Assuming fixed counts: object count may vary between examples
❌ Ignoring relative position: transform may depend on where objects are

════════════════════════════════════════════════════════════════════
DETERMINISTIC TIE-BREAKING
════════════════════════════════════════════════════════════════════

When multiple candidates are equivalent, select by priority:
  1. Top-most (minimum row index)
  2. Left-most (minimum column index)
  3. Smallest area (fewest pixels)
  4. Smallest color value

════════════════════════════════════════════════════════════════════
NUMPY PRIMITIVES REFERENCE
════════════════════════════════════════════════════════════════════

# GRID BASICS
H, W = grid.shape                          # Dimensions
out = np.zeros((H, W), dtype=int)          # Empty grid
out = np.full((H, W), fill_value=c)        # Filled with color c
out = grid.copy()                          # Copy (always use for mutation)
out = np.zeros_like(grid)                  # Same shape, all zeros

# GEOMETRIC TRANSFORMS
np.rot90(grid, k=1)                        # 90° CCW (k=1,2,3)
np.rot90(grid, k=-1)                       # 90° CW
np.flip(grid, axis=0)                      # Flip vertical
np.flip(grid, axis=1)                      # Flip horizontal
grid.T                                     # Transpose

# SLICING
grid[r, c]                                 # Single cell
grid[r1:r2, c1:c2]                         # Subgrid [r1,r2) × [c1,c2)
grid[r, :]                                 # Row r
grid[:, c]                                 # Column c
grid[::2, ::2]                             # Stride 2 (downsample)

# MASKS & BOOLEAN INDEXING
mask = (grid == color)                     # Where equals color
mask = (grid != 0)                         # Non-background
rows, cols = np.where(mask)                # Coordinates where True
coords = np.argwhere(mask)                 # Array of [r, c] pairs
grid[mask] = new_color                     # Set masked cells
np.any(mask)                               # Any True?
np.sum(mask)                               # Count True

# COLOR OPERATIONS
colors = np.unique(grid)                   # Unique colors
colors, counts = np.unique(grid, return_counts=True)
out = np.where(grid == old, new, grid)     # Replace old→new
np.isin(grid, [1, 2, 3])                   # Mask where in list

# SCALING & TILING
np.repeat(np.repeat(grid, n, axis=0), n, axis=1)  # Upscale n×
np.kron(grid, np.ones((n, n), dtype=int))         # Upscale n× (alt)
np.tile(grid, (r, c))                             # Tile r×c times
grid[::n, ::n]                                    # Downsample n×

# BOUNDING BOX
rows, cols = np.where(grid != 0)
if len(rows) > 0:
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    cropped = grid[r_min:r_max+1, c_min:c_max+1]  # +1 for inclusive

# PADDING & STACKING
np.pad(grid, pad_width=1, constant_values=0)    # Pad all sides
np.pad(grid, ((t,b), (l,r)), constant_values=0) # Asymmetric
np.vstack([a, b])                               # Stack vertically
np.hstack([a, b])                               # Stack horizontally

# CONNECTED COMPONENTS (scipy.ndimage)
from scipy import ndimage

# 4-connectivity (orthogonal neighbors only)
labeled, n = ndimage.label(grid != 0)

# 8-connectivity (include diagonals)
struct = ndimage.generate_binary_structure(2, 2)
labeled, n = ndimage.label(grid != 0, structure=struct)

# Get component k (labels are 1-indexed)
mask_k = (labeled == k)
coords_k = np.argwhere(labeled == k)

# Morphology
ndimage.binary_dilation(mask)              # Expand
ndimage.binary_erosion(mask)               # Shrink
ndimage.binary_fill_holes(mask)            # Fill interior

════════════════════════════════════════════════════════════════════
TYPE SAFETY & ERROR PREVENTION
════════════════════════════════════════════════════════════════════

⚠️ GRID VALUES MUST ALWAYS BE INTEGERS (not strings, floats, or mixed)

DEFENSIVE PATTERNS:
# Always cast to int when creating output
out = np.zeros((H, W), dtype=int)
out = grid.astype(int)

# When assigning values, ensure int type
out[r, c] = int(color)

# Safe color replacement
out = np.where(grid == old_color, new_color, grid).astype(int)

# Final output safety — ALWAYS end transform() with:
return out.astype(int)

NEVER DO:
  ❌ grid[r, c] = "1"           # String instead of int
  ❌ colors = ["0", "1", "2"]   # String list
  ❌ if color > "0":            # Comparing to string
  ❌ out = grid / 2             # Creates floats

ALWAYS DO:
  ✓ grid[r, c] = 1              # Integer literal
  ✓ colors = [0, 1, 2]          # Integer list
  ✓ if color > 0:               # Comparing to int
  ✓ out = grid // 2             # Integer division
  ✓ return out.astype(int)      # Explicit cast on return   
"""


# =============================================================================
# Formatting Helpers for Perception Data
# =============================================================================

def _format_objects_compact(objects: list[dict[str, Any]], max_objects: int = 8) -> str:
    """Format detected objects in a compact, decision-relevant way."""
    if not objects:
        return "  (no objects detected)"
    
    lines = []
    for i, obj in enumerate(objects[:max_objects]):
        obj_id = obj.get('id', i + 1)
        color = obj.get('color', 'unknown')
        shape = obj.get('shape', 'unknown')
        size = obj.get('size', '?')
        pos = obj.get('position', '')
        special = obj.get('special', '')
        
        desc = f"  #{obj_id}: {color} {shape} (size={size})"
        if pos:
            desc += f" @ {pos}"
        if special:
            desc += f" ★{special}"
        lines.append(desc)
    
    if len(objects) > max_objects:
        lines.append(f"  ... and {len(objects) - max_objects} more objects")
    
    return '\n'.join(lines)


def _format_relationships(relationships: list[dict[str, Any]], max_rels: int = 6) -> str:
    """Format object relationships for decision-making."""
    if not relationships:
        return "  (no relationships detected)"
    
    lines = []
    for rel in relationships[:max_rels]:
        rel_type = rel.get('type', 'unknown')
        
        if rel_type == 'adjacent':
            direction = rel.get('direction', '')
            lines.append(f"  • Object {rel.get('obj1')} ↔ Object {rel.get('obj2')} ({rel_type}, {direction})")
        elif rel_type == 'contained':
            lines.append(f"  • Object {rel.get('inner')} inside Object {rel.get('outer')}")
        elif rel_type == 'aligned':
            objs = rel.get('objects', [])
            axis = rel.get('axis', '')
            lines.append(f"  • Objects {objs} aligned {axis}ly")
        else:
            lines.append(f"  • {rel}")
    
    if len(relationships) > max_rels:
        lines.append(f"  ... and {len(relationships) - max_rels} more relationships")
    
    return '\n'.join(lines)


def _format_patterns_and_features(
    global_patterns: list[str] | None,
    notable_features: list[str] | None,
) -> str:
    """Format global patterns and notable features."""
    lines = []
    
    if global_patterns:
        lines.append("  Patterns:")
        for p in global_patterns[:5]:
            lines.append(f"    • {p}")
    
    if notable_features:
        lines.append("  Notable:")
        for f in notable_features[:5]:
            lines.append(f"    • {f}")
    
    return '\n'.join(lines) if lines else "  (none detected)"


def _format_delta(delta: dict[str, Any]) -> str:
    """Format transformation delta in a decision-focused way."""
    lines = []
    
    summary = delta.get('summary', '')
    if summary:
        lines.append(f"  Summary: {summary}")
    
    obj_changes = delta.get('object_changes', [])
    if obj_changes:
        lines.append("  Object Changes:")
        for change in obj_changes[:5]:
            lines.append(f"    • {change}")
    
    color_changes = delta.get('color_changes', [])
    if color_changes:
        lines.append("  Color Changes:")
        for change in color_changes[:3]:
            lines.append(f"    • {change}")
    
    structural = delta.get('structural_changes', [])
    if structural:
        lines.append("  Structural:")
        for s in structural[:3]:
            lines.append(f"    • {s}")
    
    constants = delta.get('constants', [])
    if constants:
        lines.append(f"  Constants: {', '.join(constants)}")
    
    return '\n'.join(lines) if lines else "  (no delta computed)"


def _format_hypotheses_section(hypotheses: list[dict[str, Any]], key_insight: str | None = None) -> str:
    """Format transformation hypotheses with decision priority indicators."""
    if not hypotheses:
        return ""
    
    lines = [
        "",
        "╔" + "═" * 58 + "╗",
        "║  🧠 TRANSFORMATION HYPOTHESES (Use These for Decision)      ║",
        "╚" + "═" * 58 + "╝",
    ]
    
    if key_insight:
        lines.append(f"\n💡 KEY INSIGHT: {key_insight}")
    
    lines.append("")
    
    for h in hypotheses[:5]:
        rank = h.get("rank", "?")
        conf = h.get("confidence", "?")
        rule = h.get("rule", "No rule specified")
        evidence = h.get("evidence", "")
        
        # Confidence indicator
        conf_icon = "🟢" if conf == "HIGH" else "🟡" if conf == "MEDIUM" else "🔴"
        
        lines.append(f"  {conf_icon} #{rank} [{conf}]")
        lines.append(f"     Rule: {rule}")
        if evidence:
            # Truncate long evidence
            ev_short = evidence[:150] + "..." if len(evidence) > 150 else evidence
            lines.append(f"     Evidence: {ev_short}")
        lines.append("")
    
    lines.append("  ⚠️ VERIFY: Before coding, mentally check your chosen hypothesis against ALL examples!")
    
    return '\n'.join(lines)


def _format_observations(observations: dict[str, Any]) -> str:
    """Format task-level observations from perceiver."""
    if not observations:
        return ""
    
    lines = ["", "📋 PERCEIVER OBSERVATIONS:"]
    
    common_input = observations.get('common_input_features', [])
    if common_input:
        lines.append(f"  Input patterns: {', '.join(common_input[:4])}")
    
    common_output = observations.get('common_output_features', [])
    if common_output:
        lines.append(f"  Output patterns: {', '.join(common_output[:4])}")
    
    size_pattern = observations.get('size_pattern', '')
    if size_pattern:
        lines.append(f"  Size behavior: {size_pattern}")
    
    color_changes = observations.get('color_changes', '')
    if color_changes:
        lines.append(f"  Color behavior: {color_changes}")
    
    return '\n'.join(lines)


def _format_symmetry(has_symmetry: dict[str, bool]) -> str:
    """Format symmetry information compactly."""
    active = [k for k, v in has_symmetry.items() if v]
    if not active:
        return "none"
    return ', '.join(active)


def _infer_likely_patterns(task_analysis: TaskAnalysis) -> list[str]:
    """Infer likely pattern categories based on task analysis."""
    patterns = []
    
    # Check size behavior
    if task_analysis.consistent_size_preservation:
        patterns.append("IN-PLACE (same size throughout)")
    else:
        # Check if size changes consistently
        size_changes = []
        for ex in task_analysis.train_examples:
            t = ex.transform_analysis
            if t.size_change_percent > 50:
                size_changes.append("GROW")
            elif t.size_change_percent < -50:
                size_changes.append("SHRINK")
        if size_changes:
            patterns.append(f"SIZE CHANGE ({set(size_changes)})")
    
    # Check color behavior
    if task_analysis.consistent_color_preservation:
        patterns.append("COLORS PRESERVED")
    else:
        for ex in task_analysis.train_examples:
            t = ex.transform_analysis
            if t.new_colors_introduced:
                patterns.append(f"NEW COLORS ADDED")
                break
    
    # Check shape count
    if task_analysis.consistent_shape_count:
        patterns.append("SHAPE COUNT PRESERVED")
    
    return patterns


# =============================================================================
# Main Prompt Generation
# =============================================================================

def generate_prompt(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | list[dict[str, Any]] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    key_insight: str | None = None,
    feedback: str | None = None,
    include_analysis: bool = True,
) -> str:
    """
    Generate a rich prompt for the solver with enhanced perception integration.

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Per-example perceptions (objects, relationships, patterns)
        deltas: Per-example transformation deltas
        test_perception: Perception(s) of test input(s)
        hypotheses: Ranked transformation hypotheses from perceiver
        observations: Task-level observations from perceiver
        key_insight: The key insight about the puzzle
        feedback: Optional feedback from previous failed attempt
        include_analysis: Whether to include detailed grid analysis

    Returns:
        The complete prompt string
    """
    parts = []
    train_examples = task_data['train']
    
    # Perform task-level analysis
    task_analysis = analyze_task(task_data, "current") if include_analysis else None

    # =================================================================
    # SECTION 1: Decision Support Summary (Top of Prompt)
    # =================================================================
    parts.append("╔" + "═" * 58 + "╗")
    parts.append("║          ARC PUZZLE - TRANSFORMATION TASK                ║")
    parts.append("╚" + "═" * 58 + "╝")
    
    # Quick summary box for fast decision making
    parts.append("\n┌─ QUICK SUMMARY ─────────────────────────────────────────┐")
    parts.append(f"│  Training Examples: {len(train_examples)}")
    parts.append(f"│  Test Cases: {len(task_data['test'])}")
    
    if task_analysis:
        likely = _infer_likely_patterns(task_analysis)
        if likely:
            parts.append(f"│  Likely Patterns: {', '.join(likely[:3])}")
        if task_analysis.common_hints:
            parts.append(f"│  Common Transforms: {', '.join(task_analysis.common_hints[:3])}")
    
    # Size pattern summary
    size_info = []
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        size_info.append(f"Ex{idx+1}: {inp.shape}→{out.shape}")
    parts.append(f"│  Size Changes: {' | '.join(size_info)}")
    parts.append("└─────────────────────────────────────────────────────────┘")

    # =================================================================
    # SECTION 2: Transformation Hypotheses (If Available)
    # =================================================================
    if hypotheses:
        parts.append(_format_hypotheses_section(hypotheses, key_insight))
    
    if observations:
        parts.append(_format_observations(observations))

    # =================================================================
    # SECTION 3: Training Examples with Integrated Analysis
    # =================================================================
    parts.append("\n" + "=" * 60)
    parts.append("TRAINING EXAMPLES")
    parts.append("=" * 60)

    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        parts.append(f"\n{'─'*60}")
        parts.append(f"EXAMPLE {idx + 1} of {len(train_examples)}")
        parts.append(f"{'─'*60}")

        # INPUT GRID
        parts.append(f"\n┌─ INPUT ({inp.shape[0]}×{inp.shape[1]}) ────────────────────────────────┐")
        parts.append(grid_to_text(inp))
        
        # Input analysis
        if include_analysis:
            example_analysis = analyze_example(inp, out, idx + 1)
            inp_a = example_analysis.input_analysis
            
            parts.append(f"\n  📊 Stats: {inp_a.colors_used} colors | {inp_a.total_shapes} shapes | {inp_a.fill_ratio:.0f}% filled")
            parts.append(f"  🎨 Palette: {', '.join(inp_a.color_palette)}")
            parts.append(f"  🔲 Background: {inp_a.background_color}")
            
            sym = _format_symmetry(inp_a.has_symmetry)
            if sym != "none":
                parts.append(f"  🔄 Symmetry: {sym}")
        
        # Input perception (objects, relationships)
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            
            # Handle both nested (input/output) and flat perception formats
            inp_perc = perc.get('input', perc) if 'input' in perc else perc
            
            inp_objects = inp_perc.get('objects', [])
            if inp_objects:
                parts.append(f"\n  🔍 DETECTED OBJECTS ({len(inp_objects)}):")
                parts.append(_format_objects_compact(inp_objects))
            
            inp_rels = inp_perc.get('relationships', [])
            if inp_rels:
                parts.append(f"\n  🔗 RELATIONSHIPS:")
                parts.append(_format_relationships(inp_rels))
            
            inp_patterns = inp_perc.get('global_patterns', [])
            inp_features = inp_perc.get('notable_features', [])
            if inp_patterns or inp_features:
                parts.append(f"\n  ✨ PATTERNS & FEATURES:")
                parts.append(_format_patterns_and_features(inp_patterns, inp_features))

        # OUTPUT GRID
        parts.append(f"\n┌─ OUTPUT ({out.shape[0]}×{out.shape[1]}) ───────────────────────────────┐")
        parts.append(grid_to_text(out))
        
        # Output analysis
        if include_analysis:
            out_a = example_analysis.output_analysis
            
            parts.append(f"\n  📊 Stats: {out_a.colors_used} colors | {out_a.total_shapes} shapes | {out_a.fill_ratio:.0f}% filled")
            parts.append(f"  🎨 Palette: {', '.join(out_a.color_palette)}")
            
            sym = _format_symmetry(out_a.has_symmetry)
            if sym != "none":
                parts.append(f"  🔄 Symmetry: {sym}")
        
        # Output perception (if available)
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            out_perc = perc.get('output', {})
            
            out_objects = out_perc.get('objects', [])
            if out_objects:
                parts.append(f"\n  🔍 OUTPUT OBJECTS ({len(out_objects)}):")
                parts.append(_format_objects_compact(out_objects))

        # TRANSFORMATION ANALYSIS
        if include_analysis:
            trans_a = example_analysis.transform_analysis
            parts.append(f"\n  ⚡ TRANSFORMATION:")
            
            # Size change
            if trans_a.same_size:
                parts.append(f"     Size: SAME ({inp.shape[0]}×{inp.shape[1]})")
            else:
                parts.append(f"     Size: {inp.shape} → {out.shape} ({trans_a.size_change_percent:+.0f}%)")
            
            # Color changes
            if trans_a.colors_preserved:
                parts.append(f"     Colors: PRESERVED")
            else:
                if trans_a.new_colors:
                    parts.append(f"     Colors Added: {', '.join(trans_a.new_colors)}")
                if trans_a.removed_colors:
                    parts.append(f"     Colors Removed: {', '.join(trans_a.removed_colors)}")
            
            # Shape count
            if trans_a.same_shape_count:
                parts.append(f"     Shapes: PRESERVED ({trans_a.input_shape_count})")
            else:
                parts.append(f"     Shapes: {trans_a.input_shape_count} → {trans_a.output_shape_count}")
            
            # Hints
            if trans_a.hints:
                parts.append(f"     Hints: {', '.join(trans_a.hints[:4])}")

        # Delta (if available)
        if deltas and idx < len(deltas):
            delta = deltas[idx]
            if delta.get('summary') or delta.get('object_changes'):
                parts.append(f"\n  📝 DELTA:")
                parts.append(_format_delta(delta))

    # =================================================================
    # SECTION 4: Cross-Example Pattern Summary
    # =================================================================
    if include_analysis and task_analysis:
        parts.append("\n" + "=" * 60)
        parts.append("🔎 CROSS-EXAMPLE PATTERNS (Decision Factors)")
        parts.append("=" * 60)
        
        # Invariants (things that are always true)
        invariants = []
        if task_analysis.consistent_size_preservation:
            invariants.append("✓ Size always preserved")
        if task_analysis.consistent_color_preservation:
            invariants.append("✓ Colors always preserved")
        if task_analysis.consistent_shape_count:
            invariants.append("✓ Shape count always preserved")
        
        if invariants:
            parts.append("\n  INVARIANTS (must hold in your solution):")
            for inv in invariants:
                parts.append(f"    {inv}")
        
        if task_analysis.common_hints:
            parts.append(f"\n  COMMON HINTS: {', '.join(task_analysis.common_hints)}")
        
        # Inferred patterns
        inferred = _infer_likely_patterns(task_analysis)
        if inferred:
            parts.append(f"\n  INFERRED PATTERN CATEGORIES: {', '.join(inferred)}")

    # =================================================================
    # SECTION 5: Test Input(s)
    # =================================================================
    test_inputs = task_data['test']
    n_tests = len(test_inputs)
    
    parts.append("\n" + "=" * 60)
    parts.append(f"🎯 TEST INPUT{'S' if n_tests > 1 else ''} ({n_tests} total) - APPLY YOUR RULE HERE")
    parts.append("=" * 60)
    
    for test_idx, test_case in enumerate(test_inputs):
        test_input = np.array(test_case['input'])
        
        if n_tests > 1:
            parts.append(f"\n┌─ TEST {test_idx + 1}/{n_tests} ({test_input.shape[0]}×{test_input.shape[1]}) ─────────────────────────┐")
        else:
            parts.append(f"\n┌─ TEST INPUT ({test_input.shape[0]}×{test_input.shape[1]}) ────────────────────────────┐")
        
        parts.append(grid_to_text(test_input))

        # Test input analysis
        if include_analysis:
            test_a = analyze_grid(test_input)
            
            parts.append(f"\n  📊 Stats: {test_a.colors_used} colors | {test_a.total_shapes} shapes | {test_a.fill_ratio:.0f}% filled")
            parts.append(f"  🎨 Palette: {', '.join(test_a.color_palette)}")
            parts.append(f"  🔲 Background: {test_a.background_color}")
            
            sym = _format_symmetry(test_a.has_symmetry)
            if sym != "none":
                parts.append(f"  🔄 Symmetry: {sym}")

        # Test perception (if available)
        if test_perception:
            # Handle single perception or list
            if isinstance(test_perception, list):
                tp = test_perception[test_idx] if test_idx < len(test_perception) else {}
            else:
                tp = test_perception
            
            tp_objects = tp.get('objects', [])
            if tp_objects:
                parts.append(f"\n  🔍 TEST OBJECTS ({len(tp_objects)}):")
                parts.append(_format_objects_compact(tp_objects))
            
            tp_patterns = tp.get('global_patterns', [])
            if tp_patterns:
                parts.append(f"\n  ✨ Patterns: {', '.join(tp_patterns[:3])}")

    # =================================================================
    # SECTION 6: Feedback from Previous Attempt (if any)
    # =================================================================
    if feedback:
        parts.append("\n" + "=" * 60)
        parts.append("⚠️ FEEDBACK FROM PREVIOUS ATTEMPT - FIX THESE ISSUES")
        parts.append("=" * 60)
        parts.append(feedback)

    # =================================================================
    # SECTION 7: Task Instructions
    # =================================================================
    parts.append("""

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════

1. **STUDY** the training examples above
2. **IDENTIFY** the SINGLE rule that transforms ALL inputs to outputs
3. **VERIFY** your rule mentally on each example before coding
4. **IMPLEMENT** a `transform(grid)` function

PROVIDE:
1. Brief explanation of the pattern (2-3 sentences max)
2. Python code in a ```python block with the `transform` function

Remember:
• The rule must work for ALL training examples
• Output grid dimensions must match expected output
• All values must be integers 0-9
• Always return out.astype(int)
""")

    return '\n'.join(parts)


def generate_feedback_prompt(
    original_prompt: str,
    code: str,
    feedback_messages: list[str],
    attempt_num: int,
    expected_outputs: list[np.ndarray] | None = None,
    actual_outputs: list[np.ndarray] | None = None,
) -> str:
    """
    Generate a prompt with feedback for retry attempts.

    Args:
        original_prompt: The original task prompt
        code: The code that failed
        feedback_messages: Error messages from the failed attempt
        attempt_num: Current attempt number
        expected_outputs: Expected outputs (if available)
        actual_outputs: What the code actually produced (if available)

    Returns:
        Updated prompt with feedback
    """
    feedback_parts = [
        "",
        "=" * 60,
        f"⚠️ ATTEMPT {attempt_num} FAILED - CRITICAL FEEDBACK",
        "=" * 60,
        "",
        "YOUR PREVIOUS CODE:",
        "```python",
        code,
        "```",
        "",
        "ERRORS ENCOUNTERED:",
    ]
    
    for i, msg in enumerate(feedback_messages, 1):
        feedback_parts.append(f"  {i}. {msg}")
    
    # Add visual diff if we have expected vs actual
    if expected_outputs and actual_outputs:
        feedback_parts.append("")
        feedback_parts.append("OUTPUT COMPARISON:")
        for idx, (expected, actual) in enumerate(zip(expected_outputs, actual_outputs)):
            if expected is not None and actual is not None:
                feedback_parts.append(f"\n  Example {idx + 1}:")
                feedback_parts.append(f"    Expected shape: {expected.shape}")
                feedback_parts.append(f"    Your shape:     {actual.shape if hasattr(actual, 'shape') else 'N/A'}")
                
                if hasattr(actual, 'shape') and expected.shape == actual.shape:
                    diff_count = np.sum(expected != actual)
                    total = expected.size
                    feedback_parts.append(f"    Mismatched cells: {diff_count}/{total} ({100*diff_count/total:.1f}%)")
    
    feedback_parts.extend([
        "",
        "INSTRUCTIONS:",
        "  • Carefully analyze what went wrong",
        "  • Re-read the training examples",
        "  • Verify your hypothesis against ALL examples before coding",
        "  • Provide corrected code",
        "",
    ])

    return original_prompt + '\n'.join(feedback_parts)


# =============================================================================
# Specialized Prompt Variants
# =============================================================================

def generate_minimal_prompt(task_data: dict[str, Any]) -> str:
    """
    Generate a minimal prompt with just grids and basic instructions.
    Useful for faster models or when perception data isn't available.
    """
    parts = []
    train_examples = task_data['train']
    
    parts.append("ARC PUZZLE - Find the transformation rule\n")
    
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        parts.append(f"Example {idx + 1}:")
        parts.append(f"Input ({inp.shape[0]}×{inp.shape[1]}):")
        parts.append(grid_to_text(inp))
        parts.append(f"Output ({out.shape[0]}×{out.shape[1]}):")
        parts.append(grid_to_text(out))
        parts.append("")
    
    test_input = np.array(task_data['test'][0]['input'])
    parts.append(f"Test Input ({test_input.shape[0]}×{test_input.shape[1]}):")
    parts.append(grid_to_text(test_input))
    
    parts.append("""
Task: Write a Python function `transform(grid: np.ndarray) -> np.ndarray` that implements the transformation rule.
Return only the function, using numpy/scipy.ndimage only.
""")
    
    return '\n'.join(parts)


def generate_hypothesis_verification_prompt(
    task_data: dict[str, Any],
    hypothesis: str,
    code: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Generate a prompt for verifying/refining a specific hypothesis.
    
    Args:
        task_data: The task data
        hypothesis: The hypothesis being tested
        code: The code implementing the hypothesis
        results: Results from running the code on training examples
    
    Returns:
        Prompt for verification/refinement
    """
    parts = [
        "=" * 60,
        "HYPOTHESIS VERIFICATION",
        "=" * 60,
        "",
        f"HYPOTHESIS: {hypothesis}",
        "",
        "CODE:",
        "```python",
        code,
        "```",
        "",
        "RESULTS ON TRAINING EXAMPLES:",
    ]
    
    all_correct = True
    for i, result in enumerate(results):
        is_correct = result.get('correct', False)
        if not is_correct:
            all_correct = False
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        parts.append(f"\n  Example {i + 1}: {status}")
        
        if not is_correct:
            expected_shape = result.get('expected_shape', '?')
            actual_shape = result.get('actual_shape', '?')
            parts.append(f"    Expected shape: {expected_shape}")
            parts.append(f"    Actual shape:   {actual_shape}")
            
            if result.get('error'):
                parts.append(f"    Error: {result['error']}")
            elif result.get('diff_count'):
                parts.append(f"    Mismatched cells: {result['diff_count']}")
    
    if all_correct:
        parts.append("\n✓ All examples pass! Apply to test input.")
    else:
        parts.append("\n✗ Some examples fail. Refine your hypothesis.")
        parts.append("\nProvide:")
        parts.append("1. Analysis of what's wrong")
        parts.append("2. Refined hypothesis")
        parts.append("3. Corrected code")
    
    return '\n'.join(parts)
