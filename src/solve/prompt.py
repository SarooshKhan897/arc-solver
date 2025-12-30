"""Prompt generation for the solver."""

import json
from typing import Any

import numpy as np

from src.utils.grid import grid_to_text
from src.perception.analyzer import analyze_example, analyze_grid, analyze_task
from src.perception.perceiver import format_hypotheses_for_solver


# =============================================================================
# System Prompt (matches notebook HYPOTHESIZER_SYSTEM)
# =============================================================================

SOLVER_SYSTEM = """
You solve ARC-AGI puzzles by discovering the transformation rule from input-output examples, then implementing it as Python code.

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
# Prompt Generation
# =============================================================================

def generate_prompt(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
    include_analysis: bool = True,
) -> str:
    """
    Generate a rich prompt for the solver.

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Pre-computed perceptions for each training pair
        deltas: Pre-computed deltas for each training pair
        test_perception: Perception of the test input
        hypotheses: Pre-computed transformation hypotheses (top 5)
        feedback: Optional feedback from previous failed attempt
        include_analysis: Whether to include detailed grid analysis

    Returns:
        The complete prompt string
    """
    parts = []
    train_examples = task_data['train']

    # Header
    parts.append("=" * 60)
    parts.append("ARC PUZZLE - TRAINING EXAMPLES")
    parts.append("=" * 60)

    # Training examples
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        parts.append(f"\n{'='*60}")
        parts.append(f"TRAINING PAIR {idx + 1}")
        parts.append(f"{'='*60}")

        # Grid representation
        parts.append(f"\n--- INPUT ({inp.shape[0]}x{inp.shape[1]}) ---")
        parts.append(grid_to_text(inp))

        # Enhanced grid analysis
        if include_analysis:
            example_analysis = analyze_example(inp, out, idx + 1)
            inp_a = example_analysis.input_analysis
            out_a = example_analysis.output_analysis
            trans_a = example_analysis.transform_analysis
            
            parts.append(f"\n📊 INPUT STATS:")
            parts.append(f"  Grid Size: {inp_a.rows} × {inp_a.cols} ({inp_a.total_cells} cells)")
            parts.append(f"  Colors Used: {inp_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(inp_a.color_palette)}")
            parts.append(f"  Background: {inp_a.background_color}")
            parts.append(f"  Fill Ratio: {inp_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {inp_a.shapes_by_color}")

        parts.append(f"\n--- OUTPUT ({out.shape[0]}x{out.shape[1]}) ---")
        parts.append(grid_to_text(out))

        if include_analysis:
            parts.append(f"\n📊 OUTPUT STATS:")
            parts.append(f"  Grid Size: {out_a.rows} × {out_a.cols} ({out_a.total_cells} cells)")
            parts.append(f"  Colors Used: {out_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(out_a.color_palette)}")
            parts.append(f"  Background: {out_a.background_color}")
            parts.append(f"  Fill Ratio: {out_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {out_a.shapes_by_color}")
            
            parts.append(f"\n🔄 KEY TRANSFORMATIONS:")
            parts.append(f"  Size Change: {trans_a.size_change_cells} cells ({trans_a.size_change_percent:+.1f}%)")
            parts.append(f"  New Colors Introduced: {'YES' if trans_a.new_colors_introduced else 'NO'}")
            parts.append(f"  Density Change: {trans_a.density_change_percent:+.1f}%")
            parts.append(f"  Shape Count: {trans_a.input_shape_count} → {trans_a.output_shape_count}")
            parts.append(f"  Transform Hints: {', '.join(trans_a.hints)}")

        # Add perception if available
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            if 'input' in perc and 'output' in perc:
                parts.append("\n--- PERCEPTION ANALYSIS ---")
                parts.append(f"Input objects: {len(perc['input'].get('objects', []))}")
                parts.append(f"Output objects: {len(perc['output'].get('objects', []))}")

        # Add delta if available
        if deltas and idx < len(deltas):
            delta = deltas[idx]
            parts.append("\n--- TRANSFORMATION DELTA ---")
            if delta.get('summary'):
                parts.append(f"Summary: {delta['summary']}")
            if delta.get('object_changes'):
                parts.append(f"Changes: {json.dumps(delta['object_changes'][:5], indent=2)}")

    # Cross-example pattern summary
    if include_analysis:
        task_analysis = analyze_task(task_data, "current")
        parts.append("\n" + "=" * 60)
        parts.append("📋 CROSS-EXAMPLE PATTERNS")
        parts.append("=" * 60)
        parts.append(f"  Size always preserved: {task_analysis.consistent_size_preservation}")
        parts.append(f"  Colors always preserved: {task_analysis.consistent_color_preservation}")
        parts.append(f"  Shape count preserved: {task_analysis.consistent_shape_count}")
        if task_analysis.common_hints:
            parts.append(f"  Common hints: {', '.join(task_analysis.common_hints)}")

    # Transformation hypotheses from Perceiver (if provided)
    if hypotheses:
        parts.append("\n" + format_hypotheses_for_solver(hypotheses))

    # Size pattern summary
    parts.append("\n" + "=" * 60)
    parts.append("SIZE PATTERN SUMMARY")
    parts.append("=" * 60)

    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        parts.append(f"  Example {idx+1}: {inp.shape} → {out.shape}")

    # Test inputs - ALL of them!
    test_inputs = task_data['test']
    n_tests = len(test_inputs)
    
    parts.append("\n" + "=" * 60)
    parts.append(f"TEST INPUTS ({n_tests} total)")
    parts.append("=" * 60)
    
    for test_idx, test_case in enumerate(test_inputs):
        test_input = np.array(test_case['input'])
        
        if n_tests > 1:
            parts.append(f"\n--- TEST INPUT {test_idx + 1}/{n_tests} ({test_input.shape[0]}x{test_input.shape[1]}) ---")
        else:
            parts.append(f"\n--- TEST INPUT ({test_input.shape[0]}x{test_input.shape[1]}) ---")
        parts.append(grid_to_text(test_input))

        # Test input analysis
        if include_analysis:
            test_a = analyze_grid(test_input)
            parts.append(f"\n📊 TEST INPUT {test_idx + 1} STATS:")
            parts.append(f"  Grid Size: {test_a.rows} × {test_a.cols} ({test_a.total_cells} cells)")
            parts.append(f"  Colors Used: {test_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(test_a.color_palette)}")
            parts.append(f"  Background: {test_a.background_color}")
            parts.append(f"  Fill Ratio: {test_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {test_a.shapes_by_color}")

    # Test perceptions (if provided as list)
    if test_perception:
        # Handle both single perception and list of perceptions
        if isinstance(test_perception, list):
            for i, tp in enumerate(test_perception):
                parts.append(f"\n--- TEST {i+1} PERCEPTION ---")
                parts.append(f"Objects: {len(tp.get('objects', []))}")
                if tp.get('global_patterns'):
                    parts.append(f"Patterns: {tp['global_patterns']}")
        else:
            parts.append("\n--- TEST INPUT PERCEPTION ---")
            parts.append(f"Objects: {len(test_perception.get('objects', []))}")
            if test_perception.get('global_patterns'):
                parts.append(f"Patterns: {test_perception['global_patterns']}")

    # Feedback from previous attempt
    if feedback:
        parts.append("\n" + "=" * 60)
        parts.append("⚠️ FEEDBACK FROM PREVIOUS ATTEMPT")
        parts.append("=" * 60)
        parts.append(feedback)

    # Task instructions
    parts.append("""

============================================================
YOUR TASK
============================================================

1. Study the training examples above
2. Find the SINGLE rule that transforms ALL inputs to outputs
3. Write a `transform(grid)` function that implements this rule

Provide:
1. A brief explanation of the pattern (2-3 sentences)
2. Python code in a ```python block with the `transform` function
""")

    return '\n'.join(parts)


def generate_feedback_prompt(
    original_prompt: str,
    code: str,
    feedback_messages: list[str],
    attempt_num: int,
) -> str:
    """
    Generate a prompt with feedback for retry attempts.

    Args:
        original_prompt: The original task prompt
        code: The code that failed
        feedback_messages: Error messages from the failed attempt
        attempt_num: Current attempt number

    Returns:
        Updated prompt with feedback
    """
    feedback_section = f"""

============================================================
⚠️ ATTEMPT {attempt_num} FAILED - FEEDBACK
============================================================

Your previous code:
```python
{code}
```

Errors:
{chr(10).join(feedback_messages)}

Please fix these issues and provide corrected code.
"""

    return original_prompt + feedback_section
